from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"

app = FastAPI(title="NeuronSeg Static Zarr Server")

# CORS:
# Neuroglancer may run on a different origin (e.g., neuroglancer-demo.appspot.com),
# so it needs permission to fetch Zarr metadata/chunks from this server.
# NOTE: In production, restrict allow_origins to your real viewer/site origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve everything under backend/sessions as /data/...
#
# Filesystem:
#   backend/sessions/demo/input.zarr/volumes/raw/.zarray
#
# HTTP (when uvicorn runs on HTTPS :8001 in your setup):
#   https://127.0.0.1:8001/data/demo/input.zarr/volumes/raw/.zarray
#
# (If you run on a different host/port, the origin changes accordingly.)
app.mount("/data", StaticFiles(directory=str(SESSIONS_DIR)), name="data")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "OK",
        "hint": "Try: /data/demo/input.zarr/volumes/raw/.zarray",
    }
