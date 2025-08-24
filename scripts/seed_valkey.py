
import json
import os
import redis
from datetime import datetime

def seed_valkey():
    """
    Seeds Valkey with EUR/USD data from a JSON file.
    """
    valkey_url = os.getenv("VALKEY_URL", "redis://localhost:6379/0")
    instrument = "EUR_USD"
    key = f"market:price:{instrument}"
    json_file_path = os.path.join(os.path.dirname(__file__), '..', 'eur_usd_m1_48h.json')

    try:
        r = redis.from_url(valkey_url)
        r.ping()
        print("Connected to Valkey.")
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Valkey: {e}")
        return

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} records from {json_file_path}")
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}.")
        return

    pipeline = r.pipeline()
    for record in data:
        try:
            # Original format: "2025-08-20T03:01:00.000000000Z"
            # Strip the 'Z' and the nanoseconds part
            dt_object = datetime.fromisoformat(record['time'].replace('Z', '')).replace(tzinfo=None)
            timestamp_ms = int(dt_object.timestamp() * 1000)

            candle = {
                "t": timestamp_ms,
                "o": record["open"],
                "h": record["high"],
                "l": record["low"],
                "c": record["close"]
            }
            
            pipeline.zadd(key, {json.dumps(candle): timestamp_ms})
        except (ValueError, KeyError) as e:
            print(f"Skipping record due to error: {e}. Record: {record}")
            continue
    
    try:
        pipeline.execute()
        print(f"Successfully seeded {len(data)} records into Valkey for key '{key}'.")
    except redis.exceptions.RedisError as e:
        print(f"An error occurred during Valkey pipeline execution: {e}")


if __name__ == "__main__":
    seed_valkey()
