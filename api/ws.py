"""api/ws.py — WebSocket connection manager for live event broadcasting.

Training, adversarial, and pipeline routers call ``ws_manager.broadcast(event_dict)``
to push real-time updates to every connected browser tab.

Usage::

    from api.ws import ws_manager

    # Inside a FastAPI WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    # From any async context (router, hook, Celery callback bridge)
    await ws_manager.broadcast({"type": "training.epoch", "epoch": 5, "loss": 0.42})
"""

from __future__ import annotations

from fastapi import WebSocket


class ConnectionManager:
    """Manages a pool of active WebSocket connections and broadcasts events."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict) -> None:
        """Send *data* to all connected clients. Remove dead connections."""
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


# Module-level singleton — import this from routers and hooks.
ws_manager = ConnectionManager()
