import asyncio
import json
import tempfile
from pathlib import Path
import websockets
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'scripts'))
from websocket_service import WebSocketService

async def run_test() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        service = WebSocketService('127.0.0.1', 8765, backend_url='http://localhost:8000')
        service.api_connector.engine_output_dir = output_dir
        server_task = asyncio.create_task(service.start())
        await asyncio.sleep(1)
        try:
            async with websockets.connect('ws://127.0.0.1:8765') as ws:
                await ws.send(json.dumps({"type": "subscribe", "channels": ["signals"]}))
                await asyncio.sleep(0.1)
                signal = {
                    "symbol": "EUR/USD",
                    "signal_type": "buy",
                    "confidence": 0.9,
                    "price": 1.1,
                    "timestamp": "2025-01-01T00:00:00Z"
                }
                (output_dir / "trading_signals.json").write_text(json.dumps({"signals": [signal]}))
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(message)
                if data.get("channel") != "signals":
                    return 1
        finally:
            service.stop()
            server_task.cancel()
            try:
                await server_task
            except:
                pass
    return 0

if __name__ == "__main__":
    exit(asyncio.run(run_test()))
