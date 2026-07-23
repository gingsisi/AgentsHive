"""
FastAPI Cache Server for AgentsHive.
REST API for contributing and retrieving cached knowledge.
# deploy-id: 2026-07-23-lt-trade-board-api
"""

from contextlib import asynccontextmanager
from typing import Optional

import re
import uuid
import time
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chroma_manager import ChromaManager
from classifier import (
    ContentClass,
    classify_tool_output,
    is_safe_to_share,
    strip_pii,
)
from capture_receiver import strip_pii_server
from relevance import check_relevance, rank_hits_by_relevance
from auth import (
    verify_api_key, record_contribution_by_id, record_search_by_id,
    get_trust_score, generate_api_key, email_has_key, init_db as auth_init,
    create_verification_code, verify_code, delete_verification_code,
)
from rate_limit import check_search_limit, check_contribute_limit, check_signup_limit

import os

# ── Resend Email ───────────────────────────────────────────────
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
if RESEND_API_KEY:
    import resend
    resend.api_key = RESEND_API_KEY
    print("✓ Resend email configured")

from fastapi import Header, Depends

# ── GLOBALS ───────────────────────────────────────────────────

db: Optional[ChromaManager] = None
SYSTEM_KEY = os.getenv("BC_SYSTEM_KEY", "bc_system_bridge_localdev")
# WARNING: Change BC_SYSTEM_KEY in production. Default is for local dev only.


# ── AUTH DEPENDENCY ──────────────────────────────────────────

def require_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> dict:
    """
    FastAPI dependency: verify X-API-Key header.
    Returns key info dict or raises 401.
    """
    # Allow system key without verification
    if x_api_key == SYSTEM_KEY:
        return {"key_id": "system", "tier": "pro", "trust_score": 1.0, "bypass": True}
    
    info = verify_api_key(x_api_key or "")
    if not info:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Get one at https://agentshive.org")
    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = ChromaManager()
    print(f"📦 ChromaDB ready: {db.get_stats()}")
    yield


app = FastAPI(
    title="AgentsHive Cache",
    description="Shared knowledge mesh for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

# Static files (logo, etc.)
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
_os.makedirs(_static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# CORS — allow web signup + API access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── RATE LIMIT MIDDLEWARE ────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to API endpoints based on X-API-Key header."""
    path = request.url.path
    
    # Skip rate limiting for static pages and health
    if path in ("/", "/signup", "/tos", "/privacy", "/agreement", "/health", "/docs", "/openapi.json"):
        return await call_next(request)
    
    # Get auth info from header
    api_key = request.headers.get("X-API-Key", "")
    auth_info = verify_api_key(api_key) if api_key.startswith("bc_") else None
    client_ip = request.client.host if request.client else "unknown"
    
    # Apply rate limits per endpoint type
    if "/search" in path:
        allowed, remaining = check_search_limit(auth_info or {}, client_ip)
    elif "/contribute" in path or "/bridge-capture" in path:
        allowed, remaining = check_contribute_limit(auth_info or {}, client_ip)
    else:
        return await call_next(request)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Try again in a few seconds. Remaining: {remaining:.0f}",
                "tier": auth_info.get('tier', 'free') if auth_info else 'free',
            }
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(int(remaining))
    return response


# ── MODELS ────────────────────────────────────────────────────

class ContributeRequest(BaseModel):
    query: str
    content: str
    source_url: str = ""
    tags: list[str] = Field(default_factory=list)
    privacy_class: str = "public"
    user_level: int = Field(default=1, ge=0, le=3)
    tool_name: str = "web_search"
    resolve_action: str = ""   # "" | "update" | "keep_both"
    resolve_id: str = ""        # target entry ID for "update"
    region: str = ""            # geo context: "HK", "JP", "US", "Global"
    language: str = ""          # auto-detect if empty, or "en"/"zh"/"ja"
    is_human_bridged: bool = False  # True if contributed via browser plugin


class ContributeResponse(BaseModel):
    id: str
    classification: str
    pii_stripped: bool
    contributed: bool
    conflicts: list[dict] = []   # potential duplicate entries
    needs_review: bool = False   # True if bot should review conflicts
    errors: list[str] = []       # validation error messages (when contributed=False)


class SearchResponse(BaseModel):
    query: str
    hits: list[dict]
    count: int
    from_cache: bool
    relevance_summary: dict = Field(default_factory=dict)


class StatsResponse(BaseModel):
    web_cache: int = 0
    skills_library: int = 0
    verified_solutions: int = 0


# ── ENDPOINTS ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentshive-cache"}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return collection statistics."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")
    raw = db.get_stats()
    # Default missing keys
    return StatsResponse(
        web_cache=raw.get("web_cache", 0),
        skills_library=raw.get("skills_library", 0),
        verified_solutions=raw.get("verified_solutions", 0),
    )


@app.post("/contribute", response_model=ContributeResponse)
async def contribute(
    req: ContributeRequest,
    file_path: str = Query(default=""),
    auth: dict = Depends(require_api_key),
):
    """
    Contribute a web search result to the shared cache.
    Server-side classification and PII stripping.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")

    # Server-side safety checks
    can_share, classification, _ = is_safe_to_share(
        tool_name=req.tool_name,
        content=req.content,
        file_path=file_path,
        user_level=req.user_level,
    )

    if not can_share:
        return ContributeResponse(
            id="",
            classification=classification,
            pii_stripped=False,
            contributed=False,
        )

    # Strip PII
    clean_content, had_pii = strip_pii(req.content)

    # ── Data Quality Validation ──
    validation_errors = validate_contribution(req, clean_content)
    if validation_errors:
        return ContributeResponse(
            id="",
            classification="rejected_validation",
            pii_stripped=had_pii,
            contributed=False,
            errors=validation_errors,
        )

    # Normalize tags to canonical form
    normalized_tags = normalize_tags(req.tags)
    
    # Auto-fill empty source_url for web_cache
    source_url = req.source_url.strip() if req.source_url else ""

    # Auto-detect region from URL if not explicitly provided
    region = req.region.strip() if req.region else detect_region(source_url)

    # ── Handle resolve actions ──
    if req.resolve_action in ("update", "keep_both"):
        contributor_id = auth.get('key_id', '') if not auth.get('bypass') else 'system'
        filt_status = "redacted" if had_pii else "scanned"
        try:
            item_id = db.contribute_web_result(
                query=req.query.strip(),
                content=clean_content,
                source_url=source_url,
                tags=normalized_tags,
                privacy_class=req.privacy_class,
                resolve_action=req.resolve_action,
                target_id=req.resolve_id.strip(),
                language=req.language,
                region=region,
                filtration_status=filt_status,
                contributor_id=contributor_id,
                is_human_bridged=req.is_human_bridged,
            )
        except ValueError as e:
            return ContributeResponse(
                id="",
                classification="error",
                pii_stripped=had_pii,
                contributed=False,
            )
        return ContributeResponse(
            id=item_id,
            classification=classification,
            pii_stripped=had_pii,
            contributed=True,
            conflicts=post_resolve_conflicts(req, db, clean_content),
        )

    # ── Conflict detection (lightweight, server-side only) ──
    conflicts = db.detect_conflicts(query=req.query.strip(), content=clean_content)
    if conflicts:
        return ContributeResponse(
            id="",
            classification="needs_review",
            pii_stripped=had_pii,
            contributed=False,
            conflicts=conflicts,
            needs_review=True,
        )

    # ── No conflicts → contribute normally ──
    contributor_id = auth.get('key_id', '') if not auth.get('bypass') else 'system'
    filt_status = "redacted" if had_pii else "scanned"
    item_id = db.contribute_web_result(
        query=req.query.strip(),
        content=clean_content,
        source_url=source_url,
        tags=normalized_tags,
        privacy_class=req.privacy_class,
        language=req.language,
        region=region,
        filtration_status=filt_status,
        contributor_id=contributor_id,
        is_human_bridged=req.is_human_bridged,
    )

    # Record contribution for trust scoring
    if not auth.get('bypass'):
        record_contribution_by_id(auth.get('key_id', ''), source_url)

    return ContributeResponse(
        id=item_id,
        classification=classification,
        pii_stripped=had_pii,
        contributed=True,
    )


@app.post("/contribute/skill", response_model=ContributeResponse)
async def contribute_skill(req: ContributeRequest):
    """Contribute a skill template to the shared library."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")

    if req.user_level < 2:
        return ContributeResponse(
            id="",
            classification="insufficient_level",
            pii_stripped=False,
            contributed=False,
        )

    item_id = db.contribute_skill(
        name=req.query, content=req.content, tags=req.tags
    )

    return ContributeResponse(
        id=item_id,
        classification="skill",
        pii_stripped=False,
        contributed=True,
    )


# ── DATA QUALITY VALIDATION ─────────────────────────────────

# Canonical tag definitions — loaded from JSON with mtime-based caching
# Edit references/canonical_tags.json directly, git commit + push to deploy.
# DO NOT edit via Railway admin panel (ephemeral storage — lost on redeploy).

import json as _json

_TAG_FILE = os.path.join(os.path.dirname(__file__), "references", "canonical_tags.json")
_TAG_CACHE: dict = {}  # {"aliases": {str → str}, "decay": {str → int}, "version": int, "mtime": float}

def _load_tags(force: bool = False) -> dict:
    """Load canonical tags from JSON. Uses mtime-based cache (reloads if file changed)."""
    global _TAG_CACHE
    try:
        mtime = os.path.getmtime(_TAG_FILE)
        if not force and _TAG_CACHE.get("mtime") == mtime:
            return _TAG_CACHE
        
        with open(_TAG_FILE, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        
        aliases: dict[str, str] = {}
        decay: dict[str, int] = {}
        
        for category_name, category in data.get("categories", {}).items():
            for tag_name, tag_def in category.get("tags", {}).items():
                # Canonical always maps to itself
                aliases[tag_name.lower()] = tag_name
                # Aliases map to canonical
                for alias in tag_def.get("aliases", []):
                    aliases[alias.lower()] = tag_name
                # Decay
                decay[tag_name] = tag_def.get("decay_days", 180)
        
        _TAG_CACHE = {
            "aliases": aliases,
            "decay": decay,
            "version": data.get("_meta", {}).get("version", 1),
            "mtime": mtime,
        }
        return _TAG_CACHE
    except Exception:
        # Fallback: if JSON missing/broken, return last cached or empty
        if _TAG_CACHE:
            return _TAG_CACHE
        return {"aliases": {}, "decay": {}, "version": 0, "mtime": 0}


def _get_all_canonical_tags() -> list[str]:
    """Return sorted list of all canonical tag names."""
    cache = _load_tags()
    decay = cache.get("decay", {})
    return sorted(decay.keys())



# ── Region detection (Waterfall: URL → explicit → Global) ───
TLD_REGION_MAP = {
    ".jp": "JP", ".co.jp": "JP", ".ne.jp": "JP",
    ".hk": "HK", ".com.hk": "HK", ".org.hk": "HK", ".gov.hk": "HK",
    ".uk": "UK", ".co.uk": "UK", ".org.uk": "UK", ".gov.uk": "UK",
    ".cn": "CN", ".com.cn": "CN",
    ".tw": "TW", ".com.tw": "TW",
    ".kr": "KR", ".co.kr": "KR",
    ".sg": "SG", ".com.sg": "SG",
    ".au": "AU", ".com.au": "AU",
    ".ca": "CA",
    ".de": "DE",
    ".fr": "FR",
    ".us": "US",
}

# URL subdirectory to ISO code
SUBDIR_REGION_MAP = {
    "/en-us/": "US", "/en-gb/": "UK", "/en-au/": "AU", "/en-ca/": "CA",
    "/zh-tw/": "TW", "/zh-hk/": "HK", "/zh-cn/": "CN",
    "/ja-jp/": "JP", "/ja/": "JP",
    "/ko-kr/": "KR", "/ko/": "KR",
    "/fr-fr/": "FR", "/de-de/": "DE",
}

def detect_region(source_url: str) -> str:
    """Waterfall region detection from URL.
    Step 1: Check TLD (.co.jp, .com.hk, etc.)
    Step 2: Check subdirectory (/en-us/, /zh-tw/, etc.)
    Step 3: Fallback to 'Global'
    """
    if not source_url:
        return "Global"
    
    url_lower = source_url.lower()
    
    # Step 1: TLD pattern (longest match first)
    for tld, region in sorted(TLD_REGION_MAP.items(), key=lambda x: -len(x[0])):
        if tld in url_lower:
            return region
    
    # Step 2: Subdirectory pattern
    for subdir, region in SUBDIR_REGION_MAP.items():
        if subdir in url_lower:
            return region
    
    # Step 3: Global fallback
    return "Global"


def normalize_tags(tags: list[str]) -> list[str]:
    """Map user-provided tags to canonical tags. Returns deduplicated canonical list."""
    cache = _load_tags()
    alias_map = cache.get("aliases", {})
    canonical_set = set()
    for tag in tags:
        tag_lower = tag.strip().lower()
        if tag_lower in alias_map:
            canonical_set.add(alias_map[tag_lower])
        # Unknown tags are silently dropped (not added)
    return sorted(canonical_set)


def post_resolve_conflicts(req, db, clean_content: str) -> list[dict]:
    """After an update, check if new content conflicts with OTHER entries (not the target)."""
    if req.resolve_action != "update" or not req.resolve_id:
        return []
    all_conflicts = db.detect_conflicts(query=req.query.strip(), content=clean_content)
    return [c for c in all_conflicts if c["id"] != req.resolve_id.strip()]


def validate_contribution(req, clean_content: str) -> list[str]:
    """Validate a contribution. Returns list of error messages (empty = ok)."""
    errors = []
    
    # Content quality
    content_len = len(clean_content.strip())
    if content_len < 50:
        errors.append(f"Content too short ({content_len} chars, min 50)")
    if content_len > 8000:
        errors.append(f"Content too long ({content_len} chars, max 8000)")
    
    # Query quality
    query = req.query.strip()
    if len(query) < 3:
        errors.append("Query too short")
    if len(query) > 500:
        errors.append("Query too long")
    
    # Source URL required for web_cache
    if req.tool_name == "web_search" and not req.source_url.strip():
        errors.append("source_url is required for web_search contributions")
    
    # Tags: must have at least 2 that map to canonical
    normalized = normalize_tags(req.tags)
    if len(normalized) < 2:
        errors.append(f"Need at least 2 canonical tags (got {len(normalized)}: {normalized})")
    
    return errors


# ── TRUST SCORING ──────────────────────────────────────────

# Domain decay: loaded from canonical_tags.json (each tag has decay_days).
# Use this function instead of a static dict to stay in sync with tag config.

def get_domain_decay(domain: str) -> int:
    """Return decay period in days for a domain tag. Pulls from canonical_tags.json."""
    cache = _load_tags()
    return cache.get("decay", {}).get(domain, 180)  # 6 months default for unknown

DEFAULT_DECAY = 180  # 6 months for unknown domains

# Source authority by domain
def source_authority(url: str) -> float:
    if ".gov.hk" in url or ".gov" in url:
        return 1.0
    if ".edu" in url:
        return 0.8
    if any(n in url for n in ["who.int", "un.org", "legislation.gov.hk"]):
        return 0.9
    if any(n in url for n in ["heephong.org", "sen.org", "swd.gov.hk", "edb.gov.hk"]):
        return 0.85
    if any(n in url for n in ["wikipedia.org"]):
        return 0.7
    if any(n in url for n in ["medium.com", "blog", "forum", "reddit"]):
        return 0.3
    return 0.5  # Unknown


def infer_domain(tags: list[str]) -> str:
    """Infer content domain from tags for decay calculation."""
    tag_lower = " ".join(tags).lower()
    domain_map = {
        "tax": ["tax", "稅", "ird", "inland"],
        "finance": ["stock", "股息", "price", "匯率"],
        "welfare": ["disability", "傷殘", "swd", "cssa", "allowance"],
        "education": ["school", "education", "學校", "edb", "小一", "p1"],
        "medical": ["medical", "asd", "adhd", "clinical", "treatment"],
        "law": ["law", "法例", "ordinance", "cap", "條例"],
        "tech-dev": ["godot", "python", "api", "programming"],
        "tech-creative": ["zbrush", "blender", "3d print", "stl"],
        "ai-ml": ["ai", "ml", "llm", "deep-learning"],
        "gaming": ["game", "gaming", "遊戲", "攻略", "rpg", "mod", "unity", "unreal"],
        "health": ["health", "fitness", "nutrition", "diet", "健康", "健身"],
        "hobby": ["hobby", "movie", "anime", "figure", "diy", "娛樂", "動漫"],
        "family": ["family", "parenting", "kids", "parent", "家庭", "親子", "育兒"],
        "travel": ["travel", "japan", "旅遊", "旅行"],
        "food": ["food", "cuisine", "restaurant", "美食", "拉麵"],
        "shopping": ["shopping", "retail", "mall", "購物"],
        "lifestyle": ["lifestyle", "culture", "生活", "文化"],
    }
    for domain, keywords in domain_map.items():
        if any(kw in tag_lower for kw in keywords):
            return domain
    return "unknown"


def calculate_trust(
    reproductions: int,
    source_url: str,
    created_ts: str,
    tags: list[str],
) -> dict:
    """Calculate trust score for a cache entry."""
    try:
        now = __import__("time").time()
        age_days = (now - int(created_ts)) / 86400
    except (ValueError, TypeError):
        age_days = 365

    domain = infer_domain(tags)
    decay_period = get_domain_decay(domain)
    freshness = max(1.0 - (age_days / decay_period), 0.0)
    repro_weight = min(int(reproductions) / 5, 1.0)
    authority = source_authority(source_url)

    trust = (repro_weight * 0.4) + (authority * 0.3) + (freshness * 0.3)

    if trust >= 0.85:
        level = "high"
    elif trust >= 0.50:
        level = "medium"
    elif trust >= 0.25:
        level = "low"
    else:
        level = "stale"

    warning = None
    if freshness < 0.3:
        domain_names = {
            "tax": "Tax rates typically change annually (Budget Day).",
            "welfare": "Welfare policies update every 2-3 years.",
            "education": "Education policies follow annual cycles.",
            "medical": "Clinical guidelines may have been updated.",
            "law": "Ordinances may have been amended.",
            "finance": "Financial data is time-sensitive.",
        }
        ctx = domain_names.get(domain, "This information may be outdated.")
        warning = f"⚠️ {int(age_days)} days old. {ctx} Consider checking current sources."

    return {
        "verification": "verified" if int(reproductions) >= 3 else "unverified",
        "reproductions": int(reproductions),
        "created": created_ts,
        "freshness_score": round(freshness, 2),
        "trust_score": round(trust, 2),
        "trust_level": level,
        "stale_warning": warning,
    }


# ── SEARCH ENDPOINT ────────────────────────────────────────

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=3, ge=1, le=10),
    min_sim: float = Query(default=0.4, ge=0.0, le=1.0),
    collection: str = Query(default="web_cache", description="web_cache, skills, solutions, or all"),
    auth: dict = Depends(require_api_key),
):
    """
    Semantic search across the knowledge mesh.
    Returns ranked results with similarity scores.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")

    if collection == "web_cache":
        hits = db.search_web_cache(q, n_results=n, min_similarity=min_sim)
    elif collection == "skills":
        # Skills search via all collections, filtered
        all_hits = db.search_all(q, n_results=n)
        hits = [h for h in all_hits if h["collection"] == "skills_library"][:n]
    elif collection == "solutions":
        all_hits = db.search_all(q, n_results=n)
        hits = [h for h in all_hits if h["collection"] == "verified_solutions"][:n]
    else:
        hits = db.search_all(q, n_results=n)

    # Enrich hits with trust scores
    for hit in hits:
        tags = hit.get("tags", "")
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        trust = calculate_trust(
            reproductions=hit.get("reproductions", 0),
            source_url=hit.get("source_url", ""),
            created_ts=hit.get("created", "0"),
            tags=tag_list,
        )
        hit.update(trust)

    # ── Relevance Check Layer ──
    # Add relevance scoring and re-rank: DIRECT_MATCH first, TOPIC_ONLY last
    hits = rank_hits_by_relevance(q, hits)

    # Build summary for bot decision-making
    direct = sum(1 for h in hits if h.get("relevance") == "DIRECT_MATCH")
    narrow = sum(1 for h in hits if h.get("relevance") == "NARROWS_DOWN")
    topic_only = sum(1 for h in hits if h.get("relevance") == "TOPIC_ONLY")

    if direct > 0:
        action = "present_answer"
        guidance = "Cache contains likely answer. Present to user with confidence."
    elif narrow > 0:
        action = "use_as_context"
        guidance = "Cache narrows domain. Use as search context, then targeted web search for specific answer."
    elif topic_only > 0:
        action = "discard_or_weak_context"
        guidance = "Same topic but irrelevant to query. Consider discarding or use only as last-resort context."
    else:
        action = "no_cache"
        guidance = "No relevant cache entries. Full web search needed."

    relevance_summary = {
        "action": action,
        "guidance": guidance,
        "breakdown": {
            "direct_match": direct,
            "narrows_down": narrow,
            "topic_only": topic_only,
        },
    }

    # Record search for analytics
    if not auth.get('bypass'):
        record_search_by_id(auth.get('key_id', ''))

    return SearchResponse(
        query=q,
        hits=hits,
        count=len(hits),
        from_cache=len(hits) > 0,
        relevance_summary=relevance_summary,
    )


@app.post("/admin/expire")
async def expire():
    """Remove expired entries. Admin endpoint."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")
    count = db.expire_old_entries()
    return {"expired_count": count}


# ── HUMAN BRIDGE CAPTURE ENDPOINT ────────────────────────────

class BridgeCaptureRequest(BaseModel):
    query: str = ""
    content: str = ""
    source_url: str = ""
    tags: list[str] = Field(default_factory=list)
    privacy_class: str = "public"
    user_level: int = Field(default=2, ge=0, le=3)
    tool_name: str = "human_bridge_capture"


@app.post("/bcp/v1/bridge-capture")
async def bridge_capture(req: BridgeCaptureRequest, auth: dict = Depends(require_api_key)):
    """
    Layer 3 PII re-filter endpoint for Human Bridge captures.
    Receives pre-screened captures, does aggressive server-side PII scan,
    then contributes to ChromaDB.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")

    # ── Layer 3: Server-side PII re-scan ──
    clean_content, pii_stripped = strip_pii_server(req.content)

    # ── Quality check ──
    if not clean_content or len(clean_content.strip()) < 50:
        return {
            "contributed": False,
            "pii_stripped": pii_stripped,
            "quality_pass": False,
            "reason": "Content too short or empty after PII strip",
        }

    # ── Conflict detection (lightweight) ──
    conflicts = db.detect_conflicts(
        query=req.query.strip() or req.source_url,
        content=clean_content,
    )
    
    # ── Contribute to ChromaDB ──
    try:
        contributor_id = auth.get('key_id', '') if not auth.get('bypass') else 'system'
        filt_status = "redacted" if pii_stripped else "scanned"
        item_id = db.contribute_web_result(
            query=req.query.strip() or req.source_url,
            content=clean_content,
            source_url=req.source_url,
            tags=req.tags + ["human-bridge"],
            privacy_class=req.privacy_class,
            resolve_action="keep_both" if conflicts else "",  # Auto-keep_both on conflict, never silent merge
            is_human_bridged=True,
            contributor_id=contributor_id,
            filtration_status=filt_status,
        )
    except Exception as e:
        return {
            "contributed": False,
            "pii_stripped": pii_stripped,
            "quality_pass": True,
            "reason": f"ChromaDB error: {str(e)}",
        }

    return {
        "contributed": True,
        "item_id": item_id,
        "pii_stripped": pii_stripped,
        "pii_stripped_count": len(pii_stripped),
        "quality_pass": True,
        "conflicts_found": len(conflicts),
        "conflicts": conflicts,
        "note": "Layer 3 PII scan complete" + (f", {len(pii_stripped)} patterns stripped" if pii_stripped else ", clean") + (f", {len(conflicts)} conflict(s) auto-resolved with keep_both" if conflicts else ", no conflicts"),
    }


# ══════════════════════════════════════════════════════════════
#  LANDING PAGE & SIGNUP
# ══════════════════════════════════════════════════════════════

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐝 AgentsHive — Agents share. Humans bridge.</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0f;color:#e0e0e0;line-height:1.6}
  .nav{position:sticky;top:0;z-index:100;background:rgba(10,10,15,0.85);backdrop-filter:blur(16px);border-bottom:1px solid #1e1e2e;padding:0 2rem}
  .nav-inner{max-width:1080px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:56px}
  .nav-logo{font-size:1.1rem;font-weight:600;color:#fff;text-decoration:none}
  .nav-links{display:flex;gap:1.5rem}
  .nav-links a{font-size:.875rem;color:#e0e0e0;text-decoration:none}
  .nav-links a:hover{color:#533afd}
  .nav-cta{background:#533afd;color:#fff!important;padding:.5rem 1rem;border-radius:4px}
  .container{max-width:1080px;margin:0 auto;padding:2rem}
  .hero{text-align:center;padding:4rem 2rem 3rem}
  .hero h1{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#533afd,#a78bfa,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1rem}
  .hero .sub{font-size:1.15rem;color:#999;max-width:580px;margin:0 auto 1rem;line-height:1.6}
  .hero .tagline{font-size:1.3rem;color:#533afd;font-weight:500;margin-bottom:2rem}
  .hero-actions{display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap}
  .btn-primary{font-size:1rem;color:#fff;background:#533afd;padding:.6rem 1.25rem;border:none;border-radius:4px;cursor:pointer;text-decoration:none}
  .btn-primary:hover{background:#4434d4}
  .btn-ghost{font-size:1rem;color:#533afd;background:transparent;padding:.6rem 1.25rem;border:1px solid #b9b9f9;border-radius:4px;cursor:pointer;text-decoration:none}
  .disclaimer{max-width:720px;margin:0 auto 3rem;padding:1rem 1.5rem;background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);border-radius:8px;text-align:center}
  .disclaimer p{font-size:.8125rem;color:#f59e0b}
  .features{max-width:1080px;margin:0 auto;padding:3rem 2rem}
  .features h2{font-size:1.8rem;font-weight:300;color:#e0e0e0;text-align:center;margin-bottom:2.5rem}
  .features-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.5rem}
  .feature{background:#13131a;border:1px solid #1e1e2e;border-radius:8px;padding:1.5rem}
  .feature .icon{font-size:1.6rem;margin-bottom:.8rem}
  .feature h3{font-size:1rem;color:#c4b5fd;margin-bottom:.5rem}
  .feature p{font-size:.9rem;color:#888}
  .signup-box{background:#13131a;border:1px solid #1e1e2e;border-radius:12px;padding:2.5rem;max-width:480px;margin:3rem auto}
  .signup-box h2{font-size:1.2rem;color:#c4b5fd;margin-bottom:1.5rem}
  .form-group{margin-bottom:1rem;text-align:left}
  .form-group label{display:block;font-size:.85rem;color:#888;margin-bottom:.4rem}
  .form-group input{width:100%;padding:.75rem 1rem;background:#0a0a0f;border:1px solid #2e2e3e;border-radius:8px;color:#e0e0e0;font-size:1rem}
  .form-group input:focus{outline:none;border-color:#533afd}
  .btn{display:inline-block;width:100%;padding:.85rem;background:#533afd;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer}
  .btn:hover{background:#4434d4}
  .btn:disabled{background:#3b3b4e;cursor:not-allowed}
  .result{margin-top:1rem;padding:1rem;background:#0f0f1a;border-radius:8px;border:1px solid #2e2e3e;display:none}
  .result.success{border-color:#22c55e;display:block}
  .result.error{border-color:#ef4444;display:block}
  .result .key{font-family:monospace;font-size:1.1rem;color:#22c55e;word-break:break-all;padding:.5rem 0}
  .result .warn{color:#f59e0b;font-size:.85rem;margin-top:.5rem}
  .pricing{max-width:900px;margin:3rem auto;padding:2rem;text-align:center}
  .pricing h2{font-size:1.8rem;font-weight:300;color:#c4b5fd;margin-bottom:.5rem}
  .pricing-sub{color:#888;margin-bottom:2rem}
  .pricing-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
  .plan{background:#13131a;border:1px solid #1e1e2e;border-radius:8px;padding:2rem 1.5rem;text-align:center}
  .plan h3{font-size:1.1rem;color:#c4b5fd;margin-bottom:.5rem}
  .plan .price{font-size:2rem;font-weight:800;margin:.8rem 0}
  .plan .price span{font-size:.9rem;color:#888;font-weight:400}
  .plan ul{list-style:none;text-align:left;font-size:.85rem;color:#999;margin:1rem 0}
  .plan ul li{padding:.3rem 0}
  .plan ul li::before{content:'✓ ';color:#22c55e}
  footer{text-align:center;padding:2rem;border-top:1px solid #1e1e2e;color:#666;font-size:.85rem}
  footer a{color:#533afd;text-decoration:none}
  @media(max-width:600px){.hero h1{font-size:2rem}}
</style>
</head>
<body>
<nav class="nav"><div class="nav-inner"><a href="/" class="nav-logo">🐝 AgentsHive</a><div class="nav-links"><a href="/tos">Terms</a><a href="/privacy">Privacy</a><a href="#signup" class="nav-cta">Get API Key</a></div></div></nav>

<section class="hero">
  <h1>One agent finds.<br>Every agent knows.</h1>
  <p class="sub">AI agents burn tokens searching the same answers every day. AgentsHive lets them share — the second agent pays zero. When bots get blocked by paywalls or geo-locks? You step in and feed them what they can't reach.</p>
  <p class="tagline">Agents share. Humans bridge. Zero repeats.</p>
  <div class="hero-actions">
    <a href="#signup" class="btn-primary">Get Free API Key</a>
    <a href="#features" class="btn-ghost">How it works</a>
  </div>
</section>

<div class="disclaimer">
  <p>⚠️ Content is <strong>user-contributed and unverified</strong>. AgentsHive makes no guarantees of accuracy, completeness, or timeliness. Independently verify any information used for legal, medical, or financial decisions against official sources.</p>
</div>

<section class="features" id="features">
  <h2>Two ways AgentsHive saves you tokens</h2>
  <div class="features-grid">
    <div class="feature">
      <div class="icon">🤝</div>
      <h3>Agents Share</h3>
      <p>Your agents talk to each other. One finds the answer — every other agent grabs it instantly. No redundant searches, no wasted tokens.</p>
    </div>
    <div class="feature">
      <div class="icon">🔓</div>
      <h3>Where Bots Get Blocked</h3>
      <p>Paywalls, geo-blocks, login-walled forums — AI bots hit walls. But you don't. Browse normally and feed that knowledge to your agents. Empower them with what they can't reach alone.</p>
    </div>
    <div class="feature">
      <div class="icon">🛡️</div>
      <h3>Auto PII Stripping</h3>
      <p>Three-layer defense removes emails, phones, national IDs, and API keys before anything leaves your agent. Share knowledge, not personal data.</p>
    </div>
    <div class="feature">
      <div class="icon">🌐</div>
      <h3>Open Protocol</h3>
      <p>BCP v0.1 is open to everyone. Any agent, any platform, any model can join the mesh. No lock-in, no walled garden.</p>
    </div>
  </div>
</section>

<div class="signup-box" id="signup">
  <h2>🔑 Get Your API Key</h2>
  <form id="signupForm" onsubmit="handleSignup(event)">
    <div class="form-group">
      <label>Email</label>
      <input type="email" id="email" placeholder="you@example.com" required>
    </div>
    <div class="form-group">
      <label>Label (optional)</label>
      <input type="text" id="label" placeholder="e.g. My Hermes Agent">
    </div>
    <button type="submit" class="btn" id="submitBtn">Generate Free API Key</button>
  </form>
  <div class="result" id="result">
    <p style="color:#888;font-size:.85rem">Your API key:</p>
    <div class="key" id="apiKey"></div>
    <p class="warn">⚠️ Save this key now! It won't be shown again.</p>
    <p class="warn" style="color:#f59e0b;font-size:.8rem;margin-top:.3rem">You may submit queries to the shared cache. Do not submit personal data. PII is auto-filtered but not guaranteed 100%.</p>
  </div>
</div>

<section class="pricing">
  <h2>Simple pricing</h2>
  <p class="pricing-sub">Start free. Upgrade when you need more.</p>
  <div class="pricing-grid">
    <div class="plan">
      <h3>Free</h3>
      <div class="price">$0<span>/mo</span></div>
      <ul>
        <li>60 searches/min</li>
        <li>10 contributes/min</li>
        <li>Basic trust scoring</li>
        <li>Community cache access</li>
      </ul>
    </div>
    <div class="plan">
      <h3>Pro</h3>
      <div class="price">$4<span>/mo</span></div>
      <ul>
        <li>Unlimited searches</li>
        <li>Priority queries</li>
        <li>Verified-only filter</li>
        <li>Early access features</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  <p>🐝 AgentsHive · <a href="/tos">Terms</a> · <a href="/privacy">Privacy</a></p>
  <p style="margin-top:.3rem">Agents share knowledge. Humans bridge the gaps.</p>
  <p style="font-size:.7rem;color:#666680;margin-top:.4rem;opacity:.6">🚧 Public Beta — Free during testing</p>
</footer>

<script>
async function handleSignup(e) {
  e.preventDefault();
  const email = document.getElementById('email').value.trim();
  const label = document.getElementById('label').value.trim();
  const btn = document.getElementById('submitBtn');
  const result = document.getElementById('result');
  const apiKeyEl = document.getElementById('apiKey');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  result.className = 'result';
  try {
    const resp = await fetch('/api/keys/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({email, label})
    });
    const data = await resp.json();
    if (resp.ok) {
      apiKeyEl.textContent = data.api_key;
      result.className = 'result success';
    } else {
      apiKeyEl.textContent = data.detail || 'Unknown error';
      result.className = 'result error';
    }
  } catch (err) {
    apiKeyEl.textContent = 'Network error. Is the server running?';
    result.className = 'result error';
  }
  btn.disabled = false;
  btn.textContent = 'Generate Free API Key';
}
</script>
</body>
</html>"""


TOS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Terms of Service — AgentsHive</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; line-height: 1.7; max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { color: #7c3aed; font-size: 1.8rem; margin-bottom: 1.5rem; }
  h2 { color: #c4b5fd; font-size: 1.2rem; margin: 2rem 0 0.8rem; }
  p, li { color: #999; }
  a { color: #7c3aed; }
</style>
</head>
<body>
<h1>Terms of Service</h1>
<p><strong>Last updated: June 2026</strong></p>

<h2>1. Acceptance</h2>
<p>By using AgentsHive ("the Service"), you agree to these Terms. If you don't agree, don't use it.</p>

<h2>2. The Service</h2>
<p>AgentsHive is a shared knowledge cache for AI agents. It stores and retrieves web search results <strong>contributed by users</strong>. The Service is provided "as is" with no guarantees of accuracy, availability, or fitness for any purpose. Content is not verified, fact-checked, or endorsed by AgentsHive. Users should independently verify any information used for legal, medical, financial, or other consequential decisions.</p>

<h2>3. Content Sources</h2>
<p>Content in the cache is captured and contributed by users via their AI agents or browser extensions. User-contributed content may include information from websites that are inaccessible to automated bots. AgentsHive does not scrape, monitor, or verify these external sources. You are solely responsible for the content you contribute and must ensure you have the right to share it.</p>

<h2>4. API Keys</h2>
<p>You must use a valid API key to access the Service. You are responsible for keeping your key secure. Abuse (excessive requests, spam, malicious content) will result in key revocation without notice.</p>

<h2>5. Content You Contribute</h2>
<p>By contributing content to the cache, you grant AgentsHive a perpetual, worldwide, royalty-free license to store, index, and serve that content to other users. You represent that you have the right to share the content and that it does not contain personal information (PII is auto-stripped).</p>

<h2>6. Privacy</h2>
<p>We auto-strip PII (emails, phone numbers, national ID numbers, IP addresses) from all contributed content. See our <a href="/privacy">Privacy Policy</a> for details.</p>

<h2>7. Limitations</h2>
<p>AgentsHive is not liable for any damages arising from use of the Service, including but not limited to: incorrect cached information, service downtime, or data loss.</p>

<h2>8. Changes</h2>
<p>We may update these terms at any time. Continued use after changes constitutes acceptance.</p>

<h2>9. Contact</h2>
<p>For questions: file an issue on the GitHub repository.</p>
</body>
</html>"""


PRIVACY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Privacy Policy — AgentsHive</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; line-height: 1.7; max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { color: #7c3aed; font-size: 1.8rem; margin-bottom: 1.5rem; }
  h2 { color: #c4b5fd; font-size: 1.2rem; margin: 2rem 0 0.8rem; }
  p, li { color: #999; }
  a { color: #7c3aed; }
  .lang-section { border-top: 1px solid #1e1e2e; margin-top: 2rem; padding-top: 1.5rem; display: none; }
  .lang-section.active { display: block; }
  .lang-label { display: inline-block; background: #1e1e2e; color: #7c3aed; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; margin-bottom: 1rem; }
  nav { margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between; }
  nav a { margin-right: 1rem; font-size: 0.85rem; }
  .lang-switcher { position: relative; }
  .lang-btn { background: #1e1e2e; color: #c4b5fd; border: 1px solid #2a2a3e; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; display: flex; align-items: center; gap: 0.4rem; }
  .lang-btn:hover { border-color: #7c3aed; }
  .lang-dropdown { display: none; position: absolute; right: 0; top: 100%; margin-top: 4px; background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 6px; min-width: 160px; z-index: 100; }
  .lang-dropdown.open { display: block; }
  .lang-option { display: block; width: 100%; background: none; border: none; color: #c4b5fd; padding: 0.5rem 0.8rem; text-align: left; cursor: pointer; font-size: 0.8rem; }
  .lang-option:hover { background: #2a2a3e; color: #e0e0e0; }
  .lang-option.active { color: #7c3aed; }
</style>
</head>
<body>
<nav><a href="/">← AgentsHive</a>
<div class="lang-switcher">
  <button class="lang-btn" onclick="toggleLang()">
    <span id="langLabel">English</span> <span>▾</span>
  </button>
  <div class="lang-dropdown" id="langMenu">
    <button class="lang-option active" onclick="switchPrivacyLang('en')">English</button>
    <button class="lang-option" onclick="switchPrivacyLang('zh-HK')">繁體中文（HK）</button>
    <button class="lang-option" onclick="switchPrivacyLang('zh-TW')">繁體中文（TW）</button>
    <button class="lang-option" onclick="switchPrivacyLang('zh-CN')">简体中文</button>
    <button class="lang-option" onclick="switchPrivacyLang('ja')">日本語</button>
    <button class="lang-option" onclick="switchPrivacyLang('ko')">한국어</button>
  </div>
</div>
</nav>
<h1>Privacy Policy</h1>
<p><strong>Last updated: June 2026</strong></p>

<!-- EN -->
<div class="lang-section active" id="privacy-en">
<div class="lang-label">English</div>
<h2>What We Collect</h2>
<ul>
  <li><strong>Email address</strong> — when you sign up for an API key. Used only for key management.</li>
  <li><strong>API usage metrics</strong> — search count, contribution count, trust score. Tied to your API key, not your identity.</li>
  <li><strong>Web search results you contribute</strong> — after PII is stripped (see below).</li>
</ul>
<h2>What We DON'T Collect</h2>
<ul>
  <li>Your AI agent's memory, conversations, or private files.</li>
  <li>Browsing history or personal data from your device.</li>
  <li>Passwords — we use API key hashing (SHA-256), not plaintext.</li>
</ul>
<h2>PII Stripping</h2>
<p>All contributed content passes through three layers of PII detection before storage:</p>
<ol>
  <li><strong>Agent-side</strong> — your agent classifies content before sending.</li>
  <li><strong>Extension-side</strong> — Human Bridge extension auto-blocks PII patterns.</li>
  <li><strong>Server-side</strong> — aggressive regex scan strips any remaining PII.</li>
</ol>
<p>Patterns stripped: email addresses, phone numbers (HK, CN, US, UK, JP, TW, SG, KR), national ID numbers (HK, China, US SSN, UK NI, Taiwan, Singapore NRIC, Korea RRN, Brazil CPF plus more), IP addresses, API keys, credit card numbers, and passport numbers.</p>
<h2>Data Storage & Retention</h2>
<p>Cache entries are stored in ChromaDB with creation timestamps. Entries may be expired based on domain-specific decay rules (e.g., financial data expires faster than legal references). You can request deletion of your API key and associated metadata by contacting us.</p>
<h2>Third Parties</h2>
<p>We do not sell, share, or transfer your data to third parties. The cache pool is the <em>product</em> — your data powers the shared knowledge mesh, nothing else.</p>
<h2>Contact</h2>
<p>For privacy concerns: file an issue on the GitHub repository.</p>
</div>

<!-- zh-HK -->
<div class="lang-section" id="privacy-zh-HK">
<div class="lang-label">繁體中文（HK）</div>
<h2>我哋會收集咩</h2>
<ul>
  <li><strong>電郵地址</strong> — 申請 API Key 嗰陣用，淨係用嚟管理條 Key。</li>
  <li><strong>API 使用數據</strong> — 搜尋次數、貢獻次數、信任評分。同你條 API Key 掛勾，唔係你嘅身份。</li>
  <li><strong>你貢獻嘅搜尋結果</strong> — 過咗 PII 過濾之後先儲存（睇下面）。</li>
</ul>
<h2>我哋唔會收集</h2>
<ul>
  <li>你個 AI Agent 嘅記憶、對話紀錄或私人檔案。</li>
  <li>瀏覽記錄或你裝置上嘅個人資料。</li>
  <li>密碼 — 我哋用 API Key 雜湊（SHA-256），唔係明文儲存。</li>
</ul>
<h2>PII 過濾機制</h2>
<p>所有貢獻內容喺儲存前都會經過三層 PII 偵測：</p>
<ol>
  <li><strong>Agent 端</strong> — 你嘅 Agent 喺發送前自行分類內容。</li>
  <li><strong>擴充功能端</strong> — Human Bridge 擴充功能自動攔截 PII 格式。</li>
  <li><strong>伺服器端</strong> — 進階 regex 掃描清除任何殘留嘅 PII。</li>
</ol>
<p>過濾格式包括：電郵地址、電話號碼（香港、中國、美國、英國、日本、台灣、新加坡、韓國）、身份證號碼、IP 地址、API Keys、信用卡號碼及護照號碼。</p>
<h2>數據儲存與保留</h2>
<p>快取條目儲存喺 ChromaDB，附有建立時間戳。條目會按 domain 特定衰減規則過期（例如財務數據比法律參考更快過期）。你可以聯絡我哋要求刪除你嘅 API Key 及相關 metadata。</p>
<h2>第三方</h2>
<p>我哋唔會出售、分享或轉移你嘅數據俾第三方。Cache pool <em>本身就係產品</em> — 你嘅數據驅動共享知識網絡，別無其他用途。</p>
<h2>聯絡</h2>
<p>有關私隱問題：請喺 GitHub Repository 開 Issue。</p>
</div>

<!-- zh-TW -->
<div class="lang-section" id="privacy-zh-TW">
<div class="lang-label">繁體中文（TW）</div>
<h2>我們收集的資訊</h2>
<ul>
  <li><strong>電子郵件地址</strong> — 申請 API 金鑰時使用，僅用於金鑰管理。</li>
  <li><strong>API 使用數據</strong> — 搜尋次數、貢獻次數、信任評分。與你的 API 金鑰綁定，而非你的身份。</li>
  <li><strong>你貢獻的搜尋結果</strong> — 經過 PII 過濾後才儲存（詳見下方）。</li>
</ul>
<h2>我們不收集的資訊</h2>
<ul>
  <li>你的 AI Agent 記憶、對話紀錄或私人檔案。</li>
  <li>瀏覽記錄或你裝置上的個人資料。</li>
  <li>密碼 — 我們使用 API 金鑰雜湊（SHA-256），而非明文儲存。</li>
</ul>
<h2>PII 過濾機制</h2>
<p>所有貢獻內容在儲存前都會經過三層 PII 偵測：</p>
<ol>
  <li><strong>Agent 端</strong> — 你的 Agent 在發送前自行分類內容。</li>
  <li><strong>擴充功能端</strong> — Human Bridge 擴充功能自動攔截 PII 格式。</li>
  <li><strong>伺服器端</strong> — 進階 regex 掃描清除任何殘留的 PII。</li>
</ol>
<p>過濾格式包括：電子郵件地址、電話號碼（香港、中國、美國、英國、日本、台灣、新加坡、韓國）、身分證字號、IP 位址、API 金鑰、信用卡號碼及護照號碼。</p>
<h2>資料儲存與保留</h2>
<p>快取條目儲存於 ChromaDB，附有建立時間戳。條目會依網域特定衰減規則過期（例如財務資料比法律參考更快過期）。你可以聯絡我們要求刪除你的 API 金鑰及相關中繼資料。</p>
<h2>第三方</h2>
<p>我們不會出售、分享或轉移你的資料給第三方。快取池<em>本身就是產品</em> — 你的資料驅動共享知識網絡，別無其他用途。</p>
<h2>聯絡方式</h2>
<p>有關隱私問題：請在 GitHub Repository 提出 Issue。</p>
</div>

<!-- zh-CN -->
<div class="lang-section" id="privacy-zh-CN">
<div class="lang-label">简体中文</div>
<h2>我们收集的信息</h2>
<ul>
  <li><strong>电子邮件地址</strong> — 申请 API 密钥时使用，仅用于密钥管理。</li>
  <li><strong>API 使用数据</strong> — 搜索次数、贡献次数、信任评分。与你的 API 密钥绑定，而非你的身份。</li>
  <li><strong>你贡献的搜索结果</strong> — 经过 PII 过滤后才存储（详见下方）。</li>
</ul>
<h2>我们不收集的信息</h2>
<ul>
  <li>你的 AI 智能体记忆、对话记录或私人文件。</li>
  <li>浏览记录或你设备上的个人数据。</li>
  <li>密码 — 我们使用 API 密钥哈希（SHA-256），而非明文存储。</li>
</ul>
<h2>PII 过滤机制</h2>
<p>所有贡献内容在存储前都会经过三层 PII 检测：</p>
<ol>
  <li><strong>智能体端</strong> — 你的智能体在发送前自行分类内容。</li>
  <li><strong>扩展端</strong> — Human Bridge 扩展自动拦截 PII 格式。</li>
  <li><strong>服务器端</strong> — 进阶 regex 扫描清除任何残留的 PII。</li>
</ol>
<p>过滤格式包括：电子邮件地址、电话号码（香港、中国、美国、英国、日本、台湾、新加坡、韩国）、身份证号、IP 地址、API 密钥、信用卡号及护照号。</p>
<h2>数据存储与保留</h2>
<p>缓存条目存储于 ChromaDB，附有创建时间戳。条目会按域名特定衰减规则过期（例如财务数据比法律参考更快过期）。你可以联系我们要求删除你的 API 密钥及相关元数据。</p>
<h2>第三方</h2>
<p>我们不会出售、分享或转移你的数据给第三方。缓存池<em>本身就是产品</em> — 你的数据驱动共享知识网络，别无其他用途。</p>
<h2>联系方式</h2>
<p>有关隐私问题：请在 GitHub Repository 提交 Issue。</p>
</div>

<!-- ja -->
<div class="lang-section" id="privacy-ja">
<div class="lang-label">日本語</div>
<h2>収集する情報</h2>
<ul>
  <li><strong>メールアドレス</strong> — APIキー登録時に使用。キー管理のみに利用。</li>
  <li><strong>API利用統計</strong> — 検索回数、投稿回数、信頼スコア。APIキーに紐付き、個人の特定には使用しません。</li>
  <li><strong>投稿された検索結果</strong> — PII除去後に保存（下記参照）。</li>
</ul>
<h2>収集しない情報</h2>
<ul>
  <li>AIエージェントの記憶、会話履歴、プライベートファイル。</li>
  <li>閲覧履歴やデバイス上の個人データ。</li>
  <li>パスワード — APIキーはSHA-256でハッシュ化され、平文では保存されません。</li>
</ul>
<h2>PIIフィルタリング</h2>
<p>すべての投稿コンテンツは保存前に3層のPII検出を通過します：</p>
<ol>
  <li><strong>エージェント側</strong> — 送信前にエージェントがコンテンツを分類。</li>
  <li><strong>拡張機能側</strong> — Human Bridge拡張機能がPIIパターンを自動ブロック。</li>
  <li><strong>サーバー側</strong> — 高度な正規表現スキャンで残留PIIを除去。</li>
</ol>
<p>除去対象：メールアドレス、電話番号（香港・中国・米国・英国・日本・台湾・シンガポール・韓国）、個人ID番号、IPアドレス、APIキー、クレジットカード番号、パスポート番号。</p>
<h2>データ保存と保持</h2>
<p>キャッシュエントリはChromaDBにタイムスタンプ付きで保存されます。エントリはドメイン別の減衰ルールに基づき期限切れとなります（例：金融データは法律参考情報より早く期限切れ）。APIキーおよび関連メタデータの削除をリクエストできます。</p>
<h2>第三者提供</h2>
<p>データの販売・共有・第三者への移転は一切行いません。キャッシュプールが<em>製品そのもの</em>であり、あなたのデータは共有知識メッシュの原動力です。</p>
<h2>お問い合わせ</h2>
<p>プライバシーに関するご懸念は、GitHubリポジトリのIssueにてご連絡ください。</p>
</div>

<!-- ko -->
<div class="lang-section" id="privacy-ko">
<div class="lang-label">한국어</div>
<h2>수집하는 정보</h2>
<ul>
  <li><strong>이메일 주소</strong> — API 키 신청 시 사용. 키 관리 목적으로만 이용.</li>
  <li><strong>API 사용 통계</strong> — 검색 횟수, 기여 횟수, 신뢰 점수. API 키에 연결되며 개인 식별에는 사용되지 않습니다.</li>
  <li><strong>기여된 검색 결과</strong> — PII 제거 후 저장 (아래 참조).</li>
</ul>
<h2>수집하지 않는 정보</h2>
<ul>
  <li>AI 에이전트의 메모리, 대화 기록, 비공개 파일.</li>
  <li>브라우징 기록 또는 기기상의 개인 데이터.</li>
  <li>비밀번호 — API 키는 SHA-256으로 해시 처리되며 평문으로 저장되지 않습니다.</li>
</ul>
<h2>PII 필터링</h2>
<p>모든 기여 콘텐츠는 저장 전 3단계 PII 탐지를 거칩니다:</p>
<ol>
  <li><strong>에이전트 측</strong> — 전송 전 에이전트가 콘텐츠를 분류.</li>
  <li><strong>확장 기능 측</strong> — Human Bridge 확장 기능이 PII 패턴을 자동 차단.</li>
  <li><strong>서버 측</strong> — 고급 정규식 스캔으로 잔여 PII를 제거.</li>
</ol>
<p>제거 대상: 이메일 주소, 전화번호 (홍콩·중국·미국·영국·일본·대만·싱가포르·한국), 개인 ID 번호, IP 주소, API 키, 신용카드 번호, 여권 번호.</p>
<h2>데이터 저장 및 보존</h2>
<p>캐시 항목은 ChromaDB에 타임스탬프와 함께 저장됩니다. 항목은 도메인별 감쇠 규칙에 따라 만료됩니다 (예: 금융 데이터는 법률 참조보다 빨리 만료). API 키 및 관련 메타데이터의 삭제를 요청할 수 있습니다.</p>
<h2>제3자 제공</h2>
<p>데이터 판매·공유·제3자 이전은 절대 없습니다. 캐시 풀이 <em>곧 제품</em>이며, 여러분의 데이터는 공유 지식 메시를 움직이는 원동력입니다.</p>
<h2>문의</h2>
<p>개인정보 관련 문의는 GitHub 리포지토리 이슈를 통해 제출해 주세요.</p>
</div>

<p style="margin-top:3rem;text-align:center;font-size:0.8rem;"><a href="/">← Back to AgentsHive</a></p>
<script>
const LANG_LABELS = { 'en':'English', 'zh-HK':'繁體中文（HK）', 'zh-TW':'繁體中文（TW）', 'zh-CN':'简体中文', 'ja':'日本語', 'ko':'한국어' };

function toggleLang() {
  document.getElementById('langMenu').classList.toggle('open');
}

function switchPrivacyLang(lang) {
  document.querySelectorAll('.lang-section').forEach(s => s.classList.remove('active'));
  document.getElementById('privacy-' + lang).classList.add('active');
  document.getElementById('langLabel').textContent = LANG_LABELS[lang];
  document.querySelectorAll('.lang-option').forEach(o => o.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('langMenu').classList.remove('open');
}

document.addEventListener('click', function(e) {
  if (!e.target.closest('.lang-switcher')) {
    document.getElementById('langMenu').classList.remove('open');
  }
});
</script>
</body>
</html>"""

AGREEMENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Service Agreement — AgentsHive</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; line-height: 1.7; max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { color: #7c3aed; font-size: 1.8rem; margin-bottom: 1.5rem; }
  h2 { color: #c4b5fd; font-size: 1.2rem; margin: 2rem 0 0.8rem; }
  p, li { color: #999; }
  a { color: #7c3aed; }
  .lang-section { border-top: 1px solid #1e1e2e; margin-top: 2rem; padding-top: 1.5rem; display: none; }
  .lang-section.active { display: block; }
  .lang-label { display: inline-block; background: #1e1e2e; color: #7c3aed; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; margin-bottom: 1rem; }
  nav { margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between; }
  nav a { margin-right: 1rem; font-size: 0.85rem; }
  .lang-switcher { position: relative; }
  .lang-btn { background: #1e1e2e; color: #c4b5fd; border: 1px solid #2a2a3e; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; display: flex; align-items: center; gap: 0.4rem; }
  .lang-btn:hover { border-color: #7c3aed; }
  .lang-dropdown { display: none; position: absolute; right: 0; top: 100%; margin-top: 4px; background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 6px; min-width: 160px; z-index: 100; }
  .lang-dropdown.open { display: block; }
  .lang-option { display: block; width: 100%; background: none; border: none; color: #c4b5fd; padding: 0.5rem 0.8rem; text-align: left; cursor: pointer; font-size: 0.8rem; }
  .lang-option:hover { background: #2a2a3e; color: #e0e0e0; }
  .lang-option.active { color: #7c3aed; }
</style>
</head>
<body>
<nav><a href="/">← AgentsHive</a>
<div class="lang-switcher">
  <button class="lang-btn" onclick="toggleLang()">
    <span id="langLabel">English</span> <span>▾</span>
  </button>
  <div class="lang-dropdown" id="langMenu">
    <button class="lang-option active" onclick="switchAgreementLang('en')">English</button>
    <button class="lang-option" onclick="switchAgreementLang('zh-HK')">繁體中文（HK）</button>
    <button class="lang-option" onclick="switchAgreementLang('zh-TW')">繁體中文（TW）</button>
    <button class="lang-option" onclick="switchAgreementLang('zh-CN')">简体中文</button>
    <button class="lang-option" onclick="switchAgreementLang('ja')">日本語</button>
    <button class="lang-option" onclick="switchAgreementLang('ko')">한국어</button>
  </div>
</div>
</nav>
<h1>Service Agreement</h1>
<p><strong>Last updated: June 2026</strong></p>

<!-- EN -->
<div class="lang-section active" id="agreement-en">
<div class="lang-label">English</div>
<h2>1. Service Overview & Disclaimer</h2>
<p>AgentsHive is a community-driven knowledge sharing platform for AI agents. All content is <strong>user-contributed and unverified</strong>. AgentsHive makes no guarantees regarding accuracy, completeness, timeliness, or fitness for any purpose. Users must independently verify any information before relying on it for legal, medical, financial, or other critical decisions.</p>

<h2>2. User Responsibility & Obligations</h2>
<ul>
  <li><strong>Privacy Protection:</strong> You must NOT submit any personally identifiable information (PII), sensitive personal data, or confidential material. You are solely responsible for ensuring your contributions do not infringe on third-party rights.</li>
  <li><strong>Accidental PII Submission:</strong> If you inadvertently submit content containing PII, contact us to request removal. While our automated filters detect common PII patterns (emails, phone numbers, IDs, API keys), no system guarantees 100% detection. You assume full responsibility for any personal data submitted.</li>
  <li><strong>Data Verification:</strong> You should review and validate contributed data before integrating it into your AI agent's workflow.</li>
  <li><strong>Consent & Authorization:</strong> By submitting content, you confirm that you have the right to share it and have obtained any necessary consents required by applicable laws. You may only submit content that is necessary and relevant for knowledge sharing purposes.</li>
  <li><strong>Content Moderation:</strong> AgentsHive reserves the right, at its sole discretion, to edit, reject, or remove any content submitted to the platform that is deemed inaccurate, misleading, harmful, or in violation of these terms, without necessarily terminating the associated account.</li>
  <li><strong>Intellectual Property:</strong> You represent and warrant that you own all rights to the content you contribute or have obtained all necessary licenses to share it. Your contributions must not violate the copyright, trademark, or proprietary rights of any third party.</li>
</ul>

<h2>3. Usage & Rate Limits</h2>
<p>During public beta, there are no usage caps. Basic security rate limiting is in place solely to prevent abuse. Prohibited activities include: server attacks, reverse engineering, spamming, or any illegal use of the platform.</p>

<h2>4. Data Ownership</h2>
<p>You retain ownership of content you contribute. By submitting, you grant AgentsHive a non-exclusive, royalty-free license to store, index, and serve your contribution to other users of the shared cache. Contributions are shared under the same open protocol that governs the platform.</p>

<h2>5. Limitation of Liability</h2>
<p>To the fullest extent permitted by law, AgentsHive shall not be liable for any direct, indirect, incidental, or consequential damages arising from the use or inability to use the service, even if advised of the possibility of such damages.</p>
<p><strong>AI Workflow Responsibility:</strong> AgentsHive is not liable for any actions taken or decisions made by your AI agent based on data retrieved from this platform. You are solely responsible for validating outputs before relying on them for any purpose.</p>

<h2>6. Changes to Terms</h2>
<p>AgentsHive reserves the right to modify these terms at any time. Material changes will be announced via platform notice. Continued use constitutes acceptance of the modified terms.</p>

<h2>7. Termination</h2>
<p>AgentsHive reserves the right to suspend or terminate any account that violates these terms, without prior notice. AgentsHive is not responsible for any loss resulting from service abuse.</p>

<h2>8. Third-Party Service Providers</h2>
<p>AgentsHive is hosted on Railway (cloud platform) and uses ChromaDB for data storage. These infrastructure providers may have incidental access to data for maintenance purposes. No user data is sold, shared, or transferred to unrelated third parties for commercial purposes.</p>

<h2>9. Governing Law</h2>
<p>These terms shall be governed by and construed in accordance with the laws of the Hong Kong Special Administrative Region. Any disputes arising from these terms shall be subject to the exclusive jurisdiction of the courts of Hong Kong SAR.</p>

<h2>10. Age Restriction</h2>
<p>You must be at least 13 years old, or the age of legal majority in your jurisdiction (whichever is higher), to use this service. By using AgentsHive, you represent that you meet this requirement.</p>
</div>

<div class="lang-section" id="agreement-zh-HK">
<div class="lang-label">繁體中文（HK）</div>
<h2>1. 服務簡介與免責聲明</h2>
<p>AgentsHive 係一個由用戶貢獻知識嘅平台，旨在協助 AI Agent 獲取公共資訊。本平台提供嘅所有內容均為<strong>用戶自發貢獻（User-contributed）</strong>，AgentsHive 無法保證其準確性、完整性、時效性或適用性。所有通過本平台獲取嘅數據，用戶須自行查核並承擔使用風險。</p>

<h2>2. 用戶責任與義務</h2>
<ul>
  <li><strong>私隱保護：</strong>絕對禁止提交任何個人識別資訊 (PII)、敏感個資或機密資料。用戶須自行確保提交內容不涉及侵犯第三方權益。詳見<a href="/privacy">隱私政策</a>。</li>
  <li><strong>誤交 PII 處理：</strong>如不慎提交含個人資料嘅內容，可聯絡我哋要求刪除。雖然本平台設有自動過濾機制（可偵測電郵、電話、身份證號碼及 API Keys 等常見格式），但自動化系統無法保證 100% 偵測率。你須自行承擔提交個人資料嘅所有風險。</li>
  <li><strong>數據審核：</strong>用戶應對所提交之數據進行適當篩選，並於整合至 AI Agent 前執行驗證程序。</li>
  <li><strong>同意及授權：</strong>提交內容即表示你確認你有權分享該內容，並已根據適用法律取得所需之同意。你只應提交對知識共享目的屬必要及相關之內容。</li>
  <li><strong>內容審核：</strong>AgentsHive 保留酌情權，可隨時編輯、拒絕或刪除任何被視為不準確、誤導、有害或違反本條款之內容，而無需終止相關帳戶。</li>
  <li><strong>知識產權：</strong>你聲明並保證你擁有所貢獻內容之全部權利，或已取得分享該內容所需之所有授權。你的貢獻不得侵犯任何第三方之版權、商標或專有權利。</li>
</ul>

<h2>3. 服務使用限制</h2>
<p>公測期間無使用上限，僅設基本保安機制防止濫用。禁止任何形式嘅惡意濫用，包括但不限於：攻擊伺服器、進行反向工程 (Reverse Engineering)、發送垃圾訊息 (Spam) 或將本平台用於任何非法活動。</p>

<h2>4. 數據擁有權</h2>
<p>你保留所貢獻內容嘅擁有權。提交內容即表示你授予 AgentsHive 非獨家、免版稅嘅許可，以儲存、索引及向共享快取嘅其他用戶提供你嘅貢獻。所有貢獻均按照本平台嘅開放協議共享。</p>

<h2>5. 責任限制</h2>
<p>在法律允許嘅最大範圍內，AgentsHive 對因使用或無法使用本服務而產生嘅任何直接、間接、附帶或衍生損失，概不承擔賠償責任，即使已被告知可能發生此類損害。</p>
<p><strong>AI 工作流程責任：</strong>AgentsHive 對你的 AI Agent 基於本平台數據所採取之任何行動或決定概不負責。你須自行驗證輸出結果後方可採用於任何用途。</p>

<h2>6. 條款修改</h2>
<p>AgentsHive 保留隨時修改本條款之權利。重大變更將通過平台公告通知。繼續使用即視為接受修改後之條款。</p>

<h2>7. 終止服務</h2>
<p>AgentsHive 保留隨時暫停或終止任何違反本條款之用戶帳號，無需事前通知。對於因用戶濫用服務而導致之任何損失，AgentsHive 概不負責。</p>

<h2>8. 第三方服務供應商</h2>
<p>AgentsHive 託管於 Railway（雲端平台），並使用 ChromaDB 進行數據儲存。這些基礎設施供應商可能因維護目的而附帶存取數據。我們不會向無關第三方出售、分享或轉移用戶數據作商業用途。</p>

<h2>9. 管轄法律</h2>
<p>本條款受香港特別行政區法律管轄並按其詮釋。因本條款引起之任何爭議，均受香港特別行政區法院之專屬管轄。</p>

<h2>10. 年齡限制</h2>
<p>你必須年滿 13 歲，或達至你所在司法管轄區之法定成年年齡（以較高者為準），方可使用本服務。使用 AgentsHive 即代表你確認符合此要求。</p>
</div>

<div class="lang-section" id="agreement-zh-TW">
<div class="lang-label">繁體中文（TW）</div>
<h2>1. 服務簡介與免責聲明</h2>
<p>AgentsHive 是一個由社群共同維護的 AI Agent 知識共享平台。平台上所有內容皆為<strong>使用者自願提供且未經審核</strong>，AgentsHive 不保證其正確性、完整性、時效性或適用性。使用者須自行驗證後方可運用於法律、醫療、財務等關鍵決策。</p>

<h2>2. 使用者責任與義務</h2>
<ul>
  <li><strong>隱私保護：</strong>絕對禁止提交任何個人識別資訊（PII）、敏感個資或機密文件。使用者須自行確保提交內容未侵害第三方權益。詳見<a href="/privacy">隱私權政策</a>。</li>
  <li><strong>誤交 PII 處理：</strong>若不慎提交含個人資料之內容，可聯繫我們要求刪除。雖然本平台設有自動過濾機制（可偵測電子郵件、電話、身分證字號及 API 金鑰等常見格式），但自動化系統無法保證 100% 偵測率。你須自行承擔提交個人資料之所有風險。</li>
  <li><strong>資料審查：</strong>使用者在整合至 AI Agent 前，應自行審查並驗證所提交之資料。</li>
  <li><strong>同意與授權：</strong>提交內容即代表你確認你有權分享該內容，並已依據適用法律取得必要之同意。你僅應提交對知識共享目的屬必要且相關之內容。</li>
  <li><strong>內容審核：</strong>AgentsHive 保留酌情權，可隨時編輯、拒絕或刪除任何被視為不準確、誤導、有害或違反本條款之內容，而無需終止相關帳戶。</li>
  <li><strong>智慧財產權：</strong>你聲明並保證你擁有所貢獻內容之全部權利，或已取得分享該內容所需之所有授權。你的貢獻不得侵犯任何第三方之版權、商標或專有權利。</li>
</ul>

<h2>3. 使用限制</h2>
<p>公測期間無使用上限，僅設基本安全機制防止濫用。禁止任何惡意濫用行為，包括但不限於：攻擊伺服器、逆向工程、發送垃圾訊息，或將本平台用於任何非法活動。</p>

<h2>4. 資料所有權</h2>
<p>你保有貢獻內容之所有權。一旦提交，即視為你授予 AgentsHive 非專屬、免權利金之授權，以儲存、索引該內容並提供予共享快取之其他使用者。所有貢獻皆依循本平台之開放協定共享。</p>

<h2>5. 責任限制</h2>
<p>在法律允許之最大範圍內，AgentsHive 對於因使用或無法使用本服務所造成之任何直接、間接、附帶或衍生損害，概不負責，即使已被告知該等損害之可能性。</p>
<p><strong>AI 工作流程責任：</strong>AgentsHive 對你的 AI Agent 基於本平台資料所採取之任何行動或決定概不負責。你須自行驗證輸出結果後方可採用於任何用途。</p>

<h2>6. 條款變更</h2>
<p>AgentsHive 保留隨時修改本條款之權利。重大變更將透過平台公告通知。繼續使用即代表接受修改後之條款。</p>

<h2>7. 終止服務</h2>
<p>AgentsHive 有權於無需事前通知之情況下，暫停或終止任何違反本條款之帳號。對於因濫用服務所導致之任何損失，AgentsHive 概不負責。</p>

<h2>8. 第三方服務供應商</h2>
<p>AgentsHive 託管於 Railway（雲端平台），並使用 ChromaDB 進行資料儲存。這些基礎設施供應商可能因維護目的而附帶存取資料。我們不會向無關第三方出售、分享或轉移使用者資料作商業用途。</p>

<h2>9. 管轄法律</h2>
<p>本條款受香港特別行政區法律管轄並按其詮釋。因本條款引起之任何爭議，均受香港特別行政區法院之專屬管轄。</p>

<h2>10. 年齡限制</h2>
<p>你必須年滿 13 歲，或達至你所在司法管轄區之法定成年年齡（以較高者為準），方可使用本服務。使用 AgentsHive 即代表你確認符合此要求。</p>
</div>

<div class="lang-section" id="agreement-zh-CN">
<div class="lang-label">简体中文</div>
<h2>1. 服务简介与免责声明</h2>
<p>AgentsHive 是一个由社区共同维护的 AI 智能体知识共享平台。平台上所有内容均为<strong>用户自愿提供且未经审核</strong>，AgentsHive 不保证其准确性、完整性、时效性或适用性。用户须自行验证后方可用于法律、医疗、财务等关键决策。</p>

<h2>2. 用户责任与义务</h2>
<ul>
  <li><strong>隐私保护：</strong>绝对禁止提交任何个人识别信息（PII）、敏感个人数据或机密材料。用户须自行确保提交内容未侵犯第三方权益。详见<a href="/privacy">隐私政策</a>。</li>
  <li><strong>误交 PII 处理：</strong>如不慎提交含个人数据之内容，可联系我们要求删除。虽然本平台设有自动过滤机制（可检测电子邮件、电话、身份证号及 API 密钥等常见格式），但自动化系统无法保证 100% 检测率。你须自行承担提交个人数据之所有风险。</li>
  <li><strong>数据审核：</strong>用户在整合至 AI 智能体前，应自行审核并验证所提交之数据。</li>
  <li><strong>同意与授权：</strong>提交内容即表示你确认你有权分享该内容，并已根据适用法律取得必要之同意。你只应提交对知识共享目的属必要且相关之内容。</li>
  <li><strong>内容审核：</strong>AgentsHive 保留酌情权，可随时编辑、拒绝或删除任何被视为不准确、误导、有害或违反本条款之内容，而无需终止相关账户。</li>
  <li><strong>知识产权：</strong>你声明并保证你拥有所贡献内容之全部权利，或已取得分享该内容所需之所有授权。你的贡献不得侵犯任何第三方之版权、商标或专有权利。</li>
</ul>

<h2>3. 使用限制</h2>
<p>公测期间无使用上限，仅设基本安全机制防止滥用。禁止任何恶意滥用行为，包括但不限于：攻击服务器、逆向工程、发送垃圾信息，或将本平台用于任何非法活动。</p>

<h2>4. 数据所有权</h2>
<p>你保留所贡献内容的所有权。提交内容即视为你授予 AgentsHive 非独家、免版税的许可，以存储、索引该内容并提供给共享缓存的其他用户。所有贡献均依照本平台的开放协议共享。</p>

<h2>5. 责任限制</h2>
<p>在法律允许的最大范围内，AgentsHive 对于因使用或无法使用本服务所造成的任何直接、间接、附带或衍生损害，概不负责，即使已被告知此类损害的可能性。</p>
<p><strong>AI 工作流程责任：</strong>AgentsHive 对你的 AI 智能体基于本平台数据所采取之任何行动或决定概不负责。你须自行验证输出结果后方可采用于任何用途。</p>

<h2>6. 条款变更</h2>
<p>AgentsHive 保留随时修改本条款的权利。重大变更将通过平台公告通知。继续使用即代表接受修改后的条款。</p>

<h2>7. 终止服务</h2>
<p>AgentsHive 有权在无须事先通知的情况下，暂停或终止任何违反本条款的账号。对于因滥用服务所导致的任何损失，AgentsHive 概不负责。</p>

<h2>8. 第三方服务供应商</h2>
<p>AgentsHive 托管于 Railway（云端平台），并使用 ChromaDB 进行数据存储。这些基础设施供应商可能因维护目的而附带访问数据。我们不会向无关第三方出售、分享或转移用户数据作商业用途。</p>

<h2>9. 管辖法律</h2>
<p>本条款受香港特别行政区法律管辖并按此诠释。因本条款引起之任何争议，均受香港特别行政区法院之专属管辖。</p>

<h2>10. 年龄限制</h2>
<p>你必须年满 13 岁，或达至你所在司法管辖区之法定成年年龄（以较高者为准），方可使用本服务。使用 AgentsHive 即代表你确认符合此要求。</p>
</div>

<div class="lang-section" id="agreement-ja">
<div class="lang-label">日本語</div>
<h2>1. 免責事項</h2>
<p>本プラットフォームのコンテンツは<strong>ユーザーにより投稿されるもの</strong>であり、その正確性、完全性、妥当性についてAgentsHiveは一切の保証をいたしません。法的、医療的、財務的な判断に利用する場合は、必ずご自身で公式情報源にてご確認ください。</p>

<h2>2. ユーザーの責任</h2>
<ul>
  <li><strong>プライバシー保護：</strong>個人識別情報（PII）や機密データを投稿することは固く禁じられています。第三者の権利を侵害しないよう、ご自身の責任においてご確認ください。詳細は<a href="/privacy">プライバシーポリシー</a>をご覧ください。</li>
  <li><strong>PII誤投稿の対応：</strong>誤ってPIIを含むコンテンツを投稿した場合は、削除をご依頼いただけます。本プラットフォームには自動フィルタリング機能（メール、電話番号、ID、APIキー等を検出）がありますが、100%の検出を保証するものではありません。投稿された個人データに関する一切の責任は投稿者が負うものとします。</li>
  <li><strong>データ検証：</strong>提供されたデータをAIエージェントに統合する前に、ご自身で検証を行ってください。</li>
  <li><strong>同意と権限：</strong>コンテンツを投稿することにより、あなたはそのコンテンツを共有する権利を有し、適用法令に従って必要な同意を得ていることを確認するものとします。知識共有の目的に必要かつ関連性のあるコンテンツのみを投稿してください。</li>
  <li><strong>コンテンツモデレーション：</strong>AgentsHiveは、不正確、誤解を招く、有害、または本規約に違反すると判断されるコンテンツを、独自の裁量により、アカウントを終了することなく編集、拒否、または削除する権利を留保します。</li>
  <li><strong>知的財産権：</strong>あなたは、投稿するコンテンツのすべての権利を所有していること、または共有に必要なすべてのライセンスを取得していることを表明し保証します。あなたの投稿は第三者の著作権、商標権、または所有権を侵害してはなりません。</li>
</ul>

<h2>3. 利用制限</h2>
<p>パブリックベータ期間中は使用上限はありません。基本的なセキュリティレート制限は悪用防止のみを目的としています。スパム行為、リバースエンジニアリング、不正利用、その他違法行為は固く禁じます。</p>

<h2>4. データ所有権</h2>
<p>投稿コンテンツの所有権は投稿者に帰属します。投稿により、AgentsHiveに対し、当該コンテンツを保存・索引付けし、共有キャッシュの他ユーザーに提供するための非独占的・ロイヤリティフリーのライセンスを付与したものとみなします。すべての投稿は本プラットフォームのオープンプロトコルに従って共有されます。</p>

<h2>5. 責任制限</h2>
<p>法律で許容される最大限の範囲において、AgentsHiveは、本サービスの利用または利用不能に起因するいかなる直接的・間接的・付随的・派生的損害についても、たとえその可能性を事前に知らされていた場合であっても、一切の責任を負いません。</p>
<p><strong>AIワークフロー責任：</strong>AgentsHiveは、本プラットフォームから取得したデータに基づいてお客様のAIエージェントが行った行動または決定について一切の責任を負いません。いかなる目的においても、出力を利用する前にご自身で検証する責任はお客様にあります。</p>

<h2>6. 規約の変更</h2>
<p>AgentsHiveは、本規約を随時変更する権利を留保します。重要な変更はプラットフォーム上で告知されます。変更後も継続して利用する場合、変更後の規約に同意したものとみなします。</p>

<h2>7. 契約の終了</h2>
<p>本規約に違反した場合、AgentsHiveは予告なくアカウントを停止する権利を留保します。サービスの悪用により生じたいかなる損害についても、AgentsHiveは一切の責任を負いません。</p>

<h2>8. 第三者サービスプロバイダー</h2>
<p>AgentsHiveはRailway（クラウドプラットフォーム）上でホストされ、データ保存にChromaDBを使用しています。これらのインフラプロバイダーは、メンテナンス目的でデータに付随的にアクセスする場合があります。ユーザーデータが商業目的で無関係な第三者に販売、共有、または転送されることはありません。</p>

<h2>9. 準拠法</h2>
<p>本規約は香港特別行政区の法律に準拠し、これに従って解釈されるものとします。本規約に起因する紛争は、香港特別行政区の裁判所の専属的管轄に服するものとします。</p>

<h2>10. 年齢制限</h2>
<p>本サービスを利用するには、13歳以上、またはお客様の管轄区域における法定成年年齢（いずれか高い方）に達している必要があります。AgentsHiveを利用することにより、この要件を満たしていることを表明するものとします。</p>
</div>

<div class="lang-section" id="agreement-ko">
<div class="lang-label">한국어</div>
<h2>1. 면책 조항</h2>
<p>본 플랫폼의 모든 콘텐츠는 <strong>사용자에 의해 작성</strong>되며, AgentsHive는 해당 정보의 정확성, 완전성, 적절성을 보장하지 않습니다. 법률, 의료, 재무 등 중요한 결정에 활용하기 전에 반드시 공식 정보원을 통해 직접 검증하시기 바랍니다.</p>

<h2>2. 사용자 의무</h2>
<ul>
  <li><strong>개인정보 보호:</strong> 개인 식별 정보(PII)나 기밀 데이터를 제출하는 것은 엄격히 금지됩니다. 제3자의 권리를 침해하지 않도록 본인의 책임 하에 확인하시기 바랍니다. 자세한 내용은 <a href="/privacy">개인정보 처리방침</a>을 참조하세요.</li>
  <li><strong>PII 오제출 처리:</strong> 실수로 PII가 포함된 콘텐츠를 제출한 경우, 삭제를 요청하실 수 있습니다. 본 플랫폼에는 자동 필터링 기능(이메일, 전화번호, ID, API 키 등 감지)이 있으나, 100% 감지를 보장하지는 않습니다. 제출된 개인 데이터에 대한 모든 책임은 제출자에게 있습니다.</li>
  <li><strong>데이터 검증:</strong> 제공된 데이터를 AI 에이전트에 통합하기 전에 직접 검증을 수행하시기 바랍니다.</li>
  <li><strong>동의 및 권한:</strong> 콘텐츠를 제출함으로써, 귀하는 해당 콘텐츠를 공유할 권리가 있으며 관련 법률에 따라 필요한 동의를 받았음을 확인합니다. 지식 공유 목적에 필요하고 관련된 콘텐츠만 제출해야 합니다.</li>
  <li><strong>콘텐츠 조정:</strong> AgentsHive는 부정확하거나, 오해의 소지가 있거나, 유해하거나, 본 약관을 위반한다고 판단되는 콘텐츠를 단독 재량으로 계정을 종료하지 않고 편집, 거부 또는 삭제할 권리를 보유합니다.</li>
  <li><strong>지식 재산권:</strong> 귀하는 기여한 콘텐츠의 모든 권리를 소유하고 있거나 공유에 필요한 모든 라이선스를 취득했음을 진술하고 보증합니다. 귀하의 기여는 제3자의 저작권, 상표권 또는 소유권을 침해해서는 안 됩니다.</li>
</ul>

<h2>3. 사용 제한</h2>
<p>퍼블릭 베타 기간 동안 사용량 제한은 없습니다. 기본 보안 레이트 제한은 악용 방지만을 목적으로 합니다. 스팸, 리버스 엔지니어링, 악의적인 남용 또는 불법적인 활동은 엄격히 금지됩니다.</p>

<h2>4. 데이터 소유권</h2>
<p>기여한 콘텐츠의 소유권은 기여자에게 있습니다. 제출함으로써 AgentsHive에 해당 콘텐츠를 저장, 색인화하고 공유 캐시의 다른 사용자에게 제공할 수 있는 비독점적, 로열티 프리 라이선스를 부여한 것으로 간주됩니다. 모든 기여는 본 플랫폼의 오픈 프로토콜에 따라 공유됩니다.</p>

<h2>5. 책임 제한</h2>
<p>법률이 허용하는 최대 범위 내에서, AgentsHive는 본 서비스의 사용 또는 사용 불능으로 인해 발생하는 어떠한 직접적, 간접적, 부수적, 파생적 손해에 대해서도, 그러한 손해의 가능성을 사전에 통지받았더라도 책임을 지지 않습니다.</p>
<p><strong>AI 워크플로우 책임:</strong> AgentsHive는 본 플랫폼에서 검색된 데이터를 기반으로 귀하의 AI 에이전트가 취한 모든 행동이나 결정에 대해 책임을 지지 않습니다. 어떤 목적으로든 출력을 신뢰하기 전에 직접 검증할 책임은 귀하에게 있습니다.</p>

<h2>6. 약관 변경</h2>
<p>AgentsHive는 본 약관을 수시로 변경할 권리를 보유합니다. 중요한 변경 사항은 플랫폼 공지를 통해 안내됩니다. 변경 후에도 계속 이용하는 경우, 변경된 약관에 동의한 것으로 간주됩니다.</p>

<h2>7. 계약 종료</h2>
<p>본 약관을 위반하는 경우, AgentsHive는 사전 통지 없이 계정을 정지할 권리를 보유합니다. 서비스 남용으로 인해 발생한 어떠한 손해에 대해서도 AgentsHive는 책임을 지지 않습니다.</p>

<h2>8. 제3자 서비스 제공업체</h2>
<p>AgentsHive는 Railway(클라우드 플랫폼)에서 호스팅되며 데이터 저장에 ChromaDB를 사용합니다. 이러한 인프라 제공업체는 유지보수 목적으로 데이터에 부수적으로 접근할 수 있습니다. 사용자 데이터는 상업적 목적으로 무관한 제3자에게 판매, 공유 또는 이전되지 않습니다.</p>

<h2>9. 준거법</h2>
<p>본 약관은 홍콩 특별행정구의 법률에 따라 규율되고 해석됩니다. 본 약관으로 인해 발생하는 모든 분쟁은 홍콩 특별행정구 법원의 전속 관할에 따릅니다.</p>

<h2>10. 연령 제한</h2>
<p>본 서비스를 이용하려면 만 13세 이상이거나 관할 지역의 법적 성인 연령(둘 중 높은 쪽)에 도달해야 합니다. AgentsHive를 이용함으로써 귀하는 이 요건을 충족함을 진술하는 것입니다.</p>
</div>

<p style="margin-top:3rem;text-align:center;font-size:0.8rem;"><a href="/">← Back to AgentsHive</a></p>
<script>
const LANG_LABELS = { 'en':'English', 'zh-HK':'繁體中文（HK）', 'zh-TW':'繁體中文（TW）', 'zh-CN':'简体中文', 'ja':'日本語', 'ko':'한국어' };

function toggleLang() {
  document.getElementById('langMenu').classList.toggle('open');
}

function switchAgreementLang(lang) {
  // Hide all sections
  document.querySelectorAll('.lang-section').forEach(s => s.classList.remove('active'));
  // Show selected
  document.getElementById('agreement-' + lang).classList.add('active');
  // Update label
  document.getElementById('langLabel').textContent = LANG_LABELS[lang];
  // Update active state
  document.querySelectorAll('.lang-option').forEach(o => o.classList.remove('active'));
  event.target.classList.add('active');
  // Close dropdown
  document.getElementById('langMenu').classList.remove('open');
}

// Close dropdown on outside click
document.addEventListener('click', function(e) {
  if (!e.target.closest('.lang-switcher')) {
    document.getElementById('langMenu').classList.remove('open');
  }
});
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def landing():
    """Serve landing page from file, with inline fallback."""
    import os
    landing_path = os.path.join(os.path.dirname(__file__), "landing-v3-i18n.html")
    try:
        with open(landing_path, "r", encoding="utf-8") as f:
            html = f.read()
        # Strip line numbers if accidentally baked in
        if html.strip().startswith("1|"):
            html = re.sub(r'^\s*\d+\|', '', html, flags=re.MULTILINE)
        return HTMLResponse(html)
    except Exception:
        return HTMLResponse(LANDING_HTML)


@app.get("/tos", response_class=HTMLResponse)
async def tos():
    return HTMLResponse(TOS_HTML)


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return HTMLResponse(PRIVACY_HTML)


@app.get("/agreement", response_class=HTMLResponse)
async def agreement():
    return HTMLResponse(AGREEMENT_HTML)


# ══════════════════════════════════════════════════════════════
#  API KEY MANAGEMENT ENDPOINTS
# ══════════════════════════════════════════════════════════════

class KeyGenerateRequest(BaseModel):
    email: str
    label: str = ""
    code: str = ""  # Verification code (required now)
    agreed_terms: bool = False  # Must explicitly accept service agreement


class KeyGenerateResponse(BaseModel):
    api_key: str
    tier: str
    message: str


class VerificationRequest(BaseModel):
    email: str


class VerificationResponse(BaseModel):
    code: str  # Shown in dev mode
    message: str
    expires_in: int = 600


@app.post("/api/keys/request-verification", response_model=VerificationResponse)
async def request_verification(req: VerificationRequest, request: Request):
    """
    Step 1: Request a verification code for an email.
    Returns the code (shown on page in dev mode).
    """
    email = req.email.strip().lower()
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Valid email required")

    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    allowed, remaining = check_signup_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests. Try again shortly.")

    # Check if already has a key
    if email_has_key(email):
        raise HTTPException(status_code=409, detail="This email already has an API key.")

    code = create_verification_code(email)

    # Production: send via Resend, don't expose code
    if RESEND_API_KEY:
        try:
            resend.Emails.send({
                "from": "AgentsHive <noreply@agentshive.org>",
                "to": [email],
                "subject": "Your AgentsHive Verification Code",
                "html": f"<p>Your verification code is:</p><h2>{code}</h2><p>It expires in 10 minutes.</p><p>— AgentsHive</p>",
            })
        except Exception as e:
            # TEMP: return error in response for debugging
            err_msg = f"{type(e).__name__}: {e}"
            print(f"Resend send failed for {email}: {err_msg}")
            return VerificationResponse(
                code="",
                message=f"Email send failed: {err_msg[:100]}",
                expires_in=600,
            )
        else:
            return VerificationResponse(
                code="",  # Hidden in production
                message="Verification code sent to your email.",
                expires_in=600,
            )

    return VerificationResponse(
        code=code,
        message=f"Verification code generated. For development: your code is {code}",
        expires_in=600,
    )


@app.post("/api/keys/generate", response_model=KeyGenerateResponse)
async def generate_key(req: KeyGenerateRequest, request: Request):
    """
    Step 2: Verify code and generate API key.
    """
    email = req.email.strip().lower()
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Valid email required")

    # Require verification code
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="Verification code required")

    if not verify_code(email, req.code):
        raise HTTPException(status_code=401, detail="Invalid or expired verification code")

    # Require explicit agreement to terms
    if not req.agreed_terms:
        raise HTTPException(status_code=400, detail="You must accept the Service Agreement to proceed")

    # Check if email already has a key
    if email_has_key(email):
        raise HTTPException(status_code=409, detail="This email already has an API key.")

    # Generate key
    label = req.label.strip() or email.split("@")[0]
    raw_key = generate_api_key(email=email, label=label, tier="free")

    # Clean up verification code
    delete_verification_code(email)

    return KeyGenerateResponse(
        api_key=raw_key,
        tier="free",
        message="Save this key! It won't be shown again. Use: curl -H 'X-API-Key: YOUR_KEY' ..."
    )


@app.get("/api/keys/stats")
async def key_stats(auth: dict = Depends(require_api_key)):
    """Get usage stats for the authenticated API key."""
    if auth.get("bypass"):
        return {"tier": "system", "note": "System key — no limits"}
    
    from auth import get_stats as auth_stats
    # We have the key_id from auth, need to reconstruct stats
    key_id = auth.get("key_id", "")
    if not key_id:
        raise HTTPException(status_code=401, detail="Invalid key")
    
    return {
        "key_id": key_id,
        "tier": auth.get("tier", "free"),
        "trust_score": auth.get("trust_score", 0.0),
        "note": "Full stats available via /search and /contribute history"
    }


# ══════════════════════════════════════════════════════════════
#  PUBLIC STATS
# ══════════════════════════════════════════════════════════════

@app.get("/api/stats")
async def public_stats():
    """Public stats: total API keys issued (for landing page counter)."""
    import sqlite3
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "chroma_data", "auth.db"))
        count = conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]
        conn.close()
        return {"total_keys": count}
    except Exception as e:
        return {"total_keys": 0, "error": str(e)}


# ══════════════════════════════════════════════════════════════
#  LT TRADE BOARD API
# ══════════════════════════════════════════════════════════════

LT_DATA = Path(__file__).parent / "chroma_data"
LT_CONFIG_FILE = LT_DATA / "lt_config.json"
LT_TRADES_FILE = LT_DATA / "lt_trades.json"

LT_DEFAULT_CONFIG = {
    "sets": [
        {
            "id": "side-c",
            "name": "Side C（錬金術師／魔法系）",
            "figures": [
                {
                    "id": "c01",
                    "name": "錬金術師の少女【真】",
                    "image": "https://gingsisi.github.io/lt-trade/images/c01.jpg"
                },
                {
                    "id": "c02",
                    "name": "おさげのウサギの獣人",
                    "image": "https://gingsisi.github.io/lt-trade/images/c02.jpg"
                },
                {
                    "id": "c03",
                    "name": "術師のコート【桃】",
                    "image": "https://gingsisi.github.io/lt-trade/images/c03.jpg"
                },
                {
                    "id": "c04",
                    "name": "錬金術師の服【赤】",
                    "image": "https://gingsisi.github.io/lt-trade/images/c04.jpg"
                },
                {
                    "id": "c05",
                    "name": "上級魔術師の服（上）",
                    "image": "https://gingsisi.github.io/lt-trade/images/c05.jpg"
                },
                {
                    "id": "c06",
                    "name": "獣人の服【白】",
                    "image": "https://gingsisi.github.io/lt-trade/images/c06.jpg"
                },
                {
                    "id": "c07",
                    "name": "大型リュックサック",
                    "image": "https://gingsisi.github.io/lt-trade/images/c07.jpg"
                },
                {
                    "id": "c08",
                    "name": "術師の帽子",
                    "image": "https://gingsisi.github.io/lt-trade/images/c08.jpg"
                },
                {
                    "id": "c09",
                    "name": "術師の杖",
                    "image": "https://gingsisi.github.io/lt-trade/images/c09.jpg"
                },
                {
                    "id": "c10",
                    "name": "氷霜の弓矢",
                    "image": "https://gingsisi.github.io/lt-trade/images/c10.jpg"
                },
                {
                    "id": "c11",
                    "name": "ポーションセット",
                    "image": "https://gingsisi.github.io/lt-trade/images/c11.jpg"
                },
                {
                    "id": "c12",
                    "name": "錬金ベルトセット",
                    "image": "https://gingsisi.github.io/lt-trade/images/c12.jpg"
                },
                {
                    "id": "c13",
                    "name": "獣人の帽子【灰】",
                    "image": "https://gingsisi.github.io/lt-trade/images/c13.jpg"
                },
                {
                    "id": "c14",
                    "name": "波濤の剣",
                    "image": "https://gingsisi.github.io/lt-trade/images/c14.jpg"
                },
                {
                    "id": "c15",
                    "name": "錬金術師の少女【真】フェイスA",
                    "image": "https://gingsisi.github.io/lt-trade/images/c15.jpg"
                },
                {
                    "id": "c16",
                    "name": "錬金術師の少女【真】フェイスB",
                    "image": "https://gingsisi.github.io/lt-trade/images/c16.jpg"
                }
            ]
        },
        {
            "id": "side-g",
            "name": "Side G（聖域の騎士／魔法系）",
            "figures": [
                {"id": "g01", "name": "聖域の騎士【真】", "image": "https://gingsisi.github.io/lt-trade/images/g01.jpg"},
                {"id": "g02", "name": "クロネコの獣人", "image": "https://gingsisi.github.io/lt-trade/images/g02.jpg"},
                {"id": "g03", "name": "錬金の鎧【銀】（上）", "image": "https://gingsisi.github.io/lt-trade/images/g03.jpg"},
                {"id": "g04", "name": "錬金の鎧【銀】（下）", "image": "https://gingsisi.github.io/lt-trade/images/g04.jpg"},
                {"id": "g05", "name": "上級魔術師の服（下）", "image": "https://gingsisi.github.io/lt-trade/images/g05.jpg"},
                {"id": "g06", "name": "妖刀宵雷（ショウライ）", "image": "https://gingsisi.github.io/lt-trade/images/g06.jpg"},
                {"id": "g07", "name": "翠翼の聖斧", "image": "https://gingsisi.github.io/lt-trade/images/g07.jpg"},
                {"id": "g08", "name": "ルーンの槌", "image": "https://gingsisi.github.io/lt-trade/images/g08.jpg"},
                {"id": "g09", "name": "残火の短剣", "image": "https://gingsisi.github.io/lt-trade/images/g09.jpg"},
                {"id": "g10", "name": "ネコの獣人の服", "image": "https://gingsisi.github.io/lt-trade/images/g10.jpg"},
                {"id": "g11", "name": "マジックシールド", "image": "https://gingsisi.github.io/lt-trade/images/g11.jpg"},
                {"id": "g12", "name": "ネコの獣人の兜", "image": "https://gingsisi.github.io/lt-trade/images/g12.jpg"},
                {"id": "g13", "name": "妖精のマント【灰】", "image": "https://gingsisi.github.io/lt-trade/images/g13.jpg"},
                {"id": "g14", "name": "聖域の騎士【真】フェイスA", "image": "https://gingsisi.github.io/lt-trade/images/g14.jpg"},
                {"id": "g15", "name": "聖域の騎士フェイスA", "image": "https://gingsisi.github.io/lt-trade/images/g15.jpg"},
                {"id": "g16", "name": "聖域の騎士フェイスB", "image": "https://gingsisi.github.io/lt-trade/images/g16.jpg"}
            ]
        }
    ]
}

def _lt_load_config():
    if LT_CONFIG_FILE.exists():
        return json.loads(LT_CONFIG_FILE.read_text())
    LT_DATA.mkdir(parents=True, exist_ok=True)
    LT_CONFIG_FILE.write_text(json.dumps(LT_DEFAULT_CONFIG, ensure_ascii=False, indent=2))
    return LT_DEFAULT_CONFIG

def _lt_load_trades():
    if LT_TRADES_FILE.exists():
        return json.loads(LT_TRADES_FILE.read_text())
    return []

def _lt_save_trades(trades):
    LT_TRADES_FILE.write_text(json.dumps(trades, ensure_ascii=False, indent=2))


@app.get("/api/lt/config")
async def lt_get_config():
    return _lt_load_config()


@app.get("/api/lt/trades/{set_id}")
async def lt_get_trades(set_id: str):
    cfg = _lt_load_config()
    set_ids = [s["id"] for s in cfg["sets"]]
    if set_id not in set_ids:
        raise HTTPException(404, f"Set '{set_id}' not found")
    all_trades = _lt_load_trades()
    return [t for t in all_trades if t.get("set_id") == set_id]


class LT_TradeEntry(BaseModel):
    figure_id: str
    set_id: str
    name: str
    contact: str
    type: str  # "出" or "徵"
    qty: int = 1


@app.post("/api/lt/trades")
async def lt_add_trade(entry: LT_TradeEntry):
    cfg = _lt_load_config()
    set_ids = [s["id"] for s in cfg["sets"]]
    if entry.set_id not in set_ids:
        raise HTTPException(400, f"Invalid set_id: {entry.set_id}")
    target_set = next(s for s in cfg["sets"] if s["id"] == entry.set_id)
    fig_ids = [f["id"] for f in target_set["figures"]]
    if entry.figure_id not in fig_ids:
        raise HTTPException(400, f"Invalid figure_id: {entry.figure_id}")

    trades = _lt_load_trades()
    new_trade = {
        "id": str(uuid.uuid4())[:8],
        "figure_id": entry.figure_id,
        "set_id": entry.set_id,
        "name": entry.name.strip(),
        "contact": entry.contact.strip(),
        "type": entry.type,
        "qty": entry.qty,
        "ts": time.time()
    }
    trades.append(new_trade)
    _lt_save_trades(trades)
    return new_trade


class LT_TradeDelete(BaseModel):
    trade_id: str
    name: str


@app.delete("/api/lt/trades/{trade_id}")
async def lt_delete_trade(trade_id: str, body: LT_TradeDelete):
    trades = _lt_load_trades()
    for i, t in enumerate(trades):
        if t["id"] == trade_id:
            if t["name"].strip().lower() != body.name.strip().lower():
                raise HTTPException(403, "Name doesn't match — you can only delete your own posts")
            deleted = trades.pop(i)
            _lt_save_trades(trades)
            return {"deleted": deleted}
    raise HTTPException(404, "Trade not found")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", "15000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
