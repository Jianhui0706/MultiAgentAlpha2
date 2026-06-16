from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, run_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(run_id, []).append(ws)

    def disconnect(self, run_id: str, ws: WebSocket) -> None:
        bucket = self._connections.get(run_id, [])
        if ws in bucket:
            bucket.remove(ws)

    async def broadcast(self, run_id: str, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._connections.get(run_id, [])):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(run_id, ws)

    def make_sync_callback(self, run_id: str) -> Any:
        """Return a sync callable usable as a progress_callback in workflow threads."""
        loop = asyncio.get_running_loop()

        def _cb(payload: dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(self.broadcast(run_id, payload), loop)

        return _cb


ws_manager = ConnectionManager()
