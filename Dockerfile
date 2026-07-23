# AgentHive Dockerfile
# Bump CACHE_BUST to force full rebuild when Railway caches stale files
FROM python:3.11-slim

ARG CACHE_BUST=6

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8080"]
