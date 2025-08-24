# File: /sep/scripts/seed_valkey.py
import os
import json
import redis # The redis-py client works with Valkey
from oanda_connector import OandaConnector 
from datetime import datetime

# Use your remote Valkey URL with credentials
VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")
INSTRUMENT = "EUR_USD"

def fetch_candles_from_oanda():
    """Fetches the last 48 hours of M1 candles."""
    connector = OandaConnector()
    if not connector.connected:
        raise ConnectionError(f"Failed to connect to OANDA")
    
    # OANDA's API can fetch up to 5000 candles at once. 48h of M1 is ~2880 candles.
    candles_raw = connector.get_candles(INSTRUMENT, "M1", count=2880)
    
    print(f"Fetched {len(candles_raw)} raw candles from OANDA.")
    
    # Convert to the required format
    candles_formatted = []
    for c in candles_raw:
        # Original format: "2025-08-20T03:01:00.000000000Z"
        # Replace 'Z' with '+00:00' to make it compatible with fromisoformat
        time_str = c['time'].replace('Z', '+00:00')
        # In some Python versions, fromisoformat can't handle 9 digits of precision.
        # We truncate to 6 digits (microseconds) for compatibility.
        if '.' in time_str:
            parts = time_str.split('.')
            time_str = parts[0] + '.' + parts[1][:6] + parts[1][9:]
        
        dt_object = datetime.fromisoformat(time_str)
        ts_ms = int(dt_object.timestamp() * 1000)

        candles_formatted.append({
            "t": ts_ms,
            "o": c['mid']['o'],
            "h": c['mid']['h'],
            "l": c['mid']['l'],
            "c": c['mid']['c'],
        })
    return candles_formatted

def seed_valkey():
    try:
        r = redis.from_url(VALKEY_URL)
        r.ping()
        print("Connected to Valkey.")
    except redis.exceptions.ConnectionError as e:
        print(f"Failed to connect to Valkey: {e}")
        return

    key = f"md:price:{INSTRUMENT}"
    
    try:
        candles = fetch_candles_from_oanda()
    except ConnectionError as e:
        print(e)
        return
    
    if not candles:
        print("No candles fetched, aborting seed.")
        return

    # Use a pipeline for efficient bulk insertion
    pipe = r.pipeline()
    pipe.delete(key) # Clear old data
    for candle in candles:
        member = json.dumps(candle, separators=(",", ":"))
        pipe.zadd(key, {member: candle["t"]})
    
    results = pipe.execute()
    print(f"Successfully seeded {sum(results[1:])} records into Valkey for key '{key}'.")

if __name__ == "__main__":
    seed_valkey()