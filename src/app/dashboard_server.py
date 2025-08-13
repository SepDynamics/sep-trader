import os
import json
from flask import Flask, jsonify

app = Flask(__name__)
LOG_FILE = os.environ.get("TRADE_LOG", "/app/logs/trades.json")


@app.route("/api/trades")
def api_trades():
    if not os.path.exists(LOG_FILE):
        return jsonify({"trades": []})
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/")
def index():
    data = api_trades().json
    count = len(data.get("trades", []))
    return (
        "<h1>SEP Dashboard</h1>"
        f"<p>Recorded trades: {count}</p>"
    )


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8050))
    app.run(host="0.0.0.0", port=port)
