"""Tests for the WebSocket ConnectionManager (api/ws.py).

Uses fake WebSocket objects — no real server needed.
"""

import pytest


# ---------------------------------------------------------------------------
# Fake WebSocket helpers
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Mimics the FastAPI WebSocket interface for testing."""

    def __init__(self):
        self.sent: list[dict] = []
        self.accepted: bool = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data: dict):
        self.sent.append(data)


class DeadWebSocket(FakeWebSocket):
    """A WebSocket that raises on send_json, simulating a broken connection."""

    async def send_json(self, data: dict):
        raise RuntimeError("Connection closed")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_accepts_and_registers():
    """connect() should call accept() on the websocket and add it to active_connections."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    ws = FakeWebSocket()

    await manager.connect(ws)

    assert ws.accepted is True
    assert ws in manager.active_connections
    assert len(manager.active_connections) == 1


@pytest.mark.asyncio
async def test_connect_multiple():
    """Multiple websockets can be connected simultaneously."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    ws1 = FakeWebSocket()
    ws2 = FakeWebSocket()

    await manager.connect(ws1)
    await manager.connect(ws2)

    assert len(manager.active_connections) == 2
    assert ws1 in manager.active_connections
    assert ws2 in manager.active_connections


@pytest.mark.asyncio
async def test_disconnect_removes_websocket():
    """disconnect() should remove the websocket from active_connections."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    ws = FakeWebSocket()

    await manager.connect(ws)
    assert len(manager.active_connections) == 1

    manager.disconnect(ws)
    assert len(manager.active_connections) == 0
    assert ws not in manager.active_connections


def test_disconnect_nonexistent_is_safe():
    """disconnect() on a websocket that was never connected should not raise."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    ws = FakeWebSocket()

    # Should not raise
    manager.disconnect(ws)
    assert len(manager.active_connections) == 0


@pytest.mark.asyncio
async def test_broadcast_sends_to_all():
    """broadcast() should send data to every connected client."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    ws1 = FakeWebSocket()
    ws2 = FakeWebSocket()
    ws3 = FakeWebSocket()

    await manager.connect(ws1)
    await manager.connect(ws2)
    await manager.connect(ws3)

    payload = {"type": "training.epoch", "epoch": 5, "loss": 0.42}
    await manager.broadcast(payload)

    for ws in [ws1, ws2, ws3]:
        assert len(ws.sent) == 1
        assert ws.sent[0] == payload


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections():
    """broadcast() should remove connections that raise exceptions."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    alive = FakeWebSocket()
    dead = DeadWebSocket()

    await manager.connect(alive)
    await manager.connect(dead)
    assert len(manager.active_connections) == 2

    payload = {"type": "status", "message": "ok"}
    await manager.broadcast(payload)

    # The dead connection should have been removed
    assert dead not in manager.active_connections
    assert len(manager.active_connections) == 1

    # The alive connection should still have received the message
    assert alive.sent == [payload]


@pytest.mark.asyncio
async def test_broadcast_empty_no_error():
    """broadcast() with no connections should complete without error."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    assert len(manager.active_connections) == 0

    # Should not raise
    await manager.broadcast({"type": "heartbeat"})


@pytest.mark.asyncio
async def test_broadcast_multiple_dead_connections():
    """broadcast() should remove all dead connections in a single pass."""
    from api.ws import ConnectionManager

    manager = ConnectionManager()
    alive = FakeWebSocket()
    dead1 = DeadWebSocket()
    dead2 = DeadWebSocket()

    await manager.connect(alive)
    await manager.connect(dead1)
    await manager.connect(dead2)

    await manager.broadcast({"type": "ping"})

    assert len(manager.active_connections) == 1
    assert alive in manager.active_connections
    assert dead1 not in manager.active_connections
    assert dead2 not in manager.active_connections


def test_module_level_singleton():
    """The module should export a ws_manager singleton instance."""
    from api.ws import ws_manager, ConnectionManager

    assert isinstance(ws_manager, ConnectionManager)


def test_singleton_is_same_instance():
    """Importing ws_manager multiple times should return the same object."""
    from api.ws import ws_manager as mgr1
    from api.ws import ws_manager as mgr2

    assert mgr1 is mgr2
