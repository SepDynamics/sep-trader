import asyncio
import json
import logging
import os
import threading
from typing import Set

import websockets

logger = logging.getLogger(__name__)

CLIENTS: Set[websockets.WebSocketServerProtocol] = set()
METRICS_PATH = os.environ.get("METRICS_PATH", "/app/data/quantum_metrics.json")

async def _load_metrics() -> dict:
    """Load quantum metrics from shared JSON source."""
    def _read_file():
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    try:
        return await asyncio.to_thread(_read_file)
    except Exception as exc:
        logger.error(f"Failed to read metrics: {exc}")
        return {}

async def _broadcast_loop(interval: float = 5.0) -> None:
    """Periodically broadcast metrics to connected clients."""
    while True:
        metrics = await _load_metrics()
        if CLIENTS and metrics:
            message = json.dumps(metrics)
            await asyncio.gather(*(ws.send(message) for ws in list(CLIENTS)))
        await asyncio.sleep(interval)

async def _ws_handler(websocket: websockets.WebSocketServerProtocol, path: str) -> None:
    if path != "/ws":
        await websocket.close()
        return
    CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(websocket)

async def _run_server(port: int) -> None:
    async with websockets.serve(_ws_handler, "0.0.0.0", port):
        await _broadcast_loop()

def start_websocket_server(port: int = 8765) -> threading.Thread:
    """Start the websocket server in a background thread."""
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_server(port))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
