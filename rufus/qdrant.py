"""
Shared Qdrant client singleton.

Connection priority
-------------------
1. Qdrant HTTP server on localhost:6333  — preferred (server manages memory)
2. Local file mode                       — fallback (loads ~8 GB into RAM for
                                           the 1.2M-product catalog; avoid on
                                           machines with <16 GB free RAM)

Start the server before launching the app:
  PowerShell:  .\\scripts\\start_qdrant.ps1
  Docker:      docker compose up -d qdrant
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

_client: QdrantClient | None = None
_LOCAL_PATH = Path("data/qdrant_storage")
_SERVER_HOST = "localhost"
_SERVER_PORT = 6333


def _try_server() -> QdrantClient | None:
    try:
        c = QdrantClient(
            host=_SERVER_HOST, port=_SERVER_PORT, grpc_port=6334,
            prefer_grpc=True, timeout=30,
        )
        c.get_collections()
        return c
    except Exception:
        return None


def get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client

    server = _try_server()
    if server is not None:
        print(f"[qdrant] Connected to server at {_SERVER_HOST}:{_SERVER_PORT}")
        _client = server
    else:
        print(
            "[qdrant] Server not found — using local file mode. "
            "RAM usage will be high (~8 GB). "
            "Start the server with: .\\scripts\\start_qdrant.ps1"
        )
        _LOCAL_PATH.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(_LOCAL_PATH))

    return _client
