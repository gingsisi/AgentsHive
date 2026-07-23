import os, uvicorn
os.environ.setdefault("PORT", "8080")
from server import app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]), log_level="info")
