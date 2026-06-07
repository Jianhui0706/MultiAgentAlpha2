from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.ws_manager import ws_manager

router = APIRouter()


@router.websocket("/ws/runs/{run_id}")
async def run_progress_ws(run_id: str, ws: WebSocket) -> None:
    await ws_manager.connect(run_id, ws)
    try:
        while True:
            # Keep connection alive; client sends pings if needed
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, ws)
