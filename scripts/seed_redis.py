#!/usr/bin/env python3
"""
Seed Redis/Valkey with sample market data for SEP Trading System
This script populates the database with realistic EUR_USD candlestick data
"""

import redis
import json
import os
import sys
from datetime import datetime, timedelta
import random
import time

def generate_realistic_candle_data(start_price=1.0850, num_candles=500, interval_minutes=5):
    """Generate realistic EUR_USD candlestick data"""
    candles = []
    current_price = start_price
    current_time = datetime.now()
    
    # Start from 48 hours ago to ensure we have recent data
    current_time = current_time - timedelta(hours=48)
    
    for i in range(num_candles):
        # Generate timestamp in milliseconds
        timestamp_ms = int(current_time.timestamp() * 1000)
        
        # Simulate realistic EUR/USD price movement
        # EUR/USD typically moves in small increments
        price_change = random.uniform(-0.0020, 0.0020)  # ±20 pips max
        new_price = current_price + price_change
        
        # Ensure price stays in realistic range (1.05 to 1.15)
        new_price = max(1.0500, min(1.1500, new_price))
        
        # Generate OHLC for this candle
        open_price = current_price
        close_price = new_price
        
        # High is typically close to max of open/close + some spread
        high_spread = random.uniform(0.0001, 0.0015)  # 1-15 pips
        high_price = max(open_price, close_price) + high_spread
        
        # Low is typically close to min of open/close - some spread  
        low_spread = random.uniform(0.0001, 0.0015)  # 1-15 pips
        low_price = min(open_price, close_price) - low_spread
        
        # Create candle data in the exact format expected by the frontend
        candle = {
            "t": timestamp_ms,
            "o": round(open_price, 5),
            "h": round(high_price, 5), 
            "l": round(low_price, 5),
            "c": round(close_price, 5)
        }
        
        candles.append(candle)
        current_price = new_price
        current_time += timedelta(minutes=interval_minutes)
    
    return candles

def seed_redis_data():
    """Seed Redis with market data"""
    # Connect to Redis
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()
        print("✅ Connected to Redis successfully")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return False
    
    # Generate sample data for EUR_USD
    print("📊 Generating realistic EUR_USD market data...")
    candles = generate_realistic_candle_data(start_price=1.0850, num_candles=500)
    
    # Store data in Redis using the exact format expected by the API
    key = "market:price:EUR_USD"
    
    try:
        # Clear existing data
        r.delete(key)
        print(f"🧹 Cleared existing data for {key}")
        
        # Add each candle as a JSON string to the sorted set
        # Use timestamp as the score for time-based querying
        stored_count = 0
        for candle in candles:
            candle_json = json.dumps(candle)
            timestamp_score = candle["t"]  # Use timestamp as score
            r.zadd(key, {candle_json: timestamp_score})
            stored_count += 1
        
        print(f"✅ Stored {stored_count} candles for EUR_USD")
        
        # Verify the data
        total_candles = r.zcard(key)
        print(f"📈 Total candles in Redis: {total_candles}")
        
        # Show a sample of the data
        sample_data = r.zrange(key, -5, -1)  # Last 5 candles
        print("\n📋 Sample data (last 5 candles):")
        for i, candle_json in enumerate(sample_data, 1):
            candle = json.loads(candle_json)
            dt = datetime.fromtimestamp(candle["t"] / 1000)
            print(f"  {i}. {dt}: O={candle['o']}, H={candle['h']}, L={candle['l']}, C={candle['c']}")
        
        # Test a time range query (like the API does)
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - (48 * 3600 * 1000)  # 48 hours ago
        
        recent_candles = r.zrangebyscore(key, from_ms, now_ms)
        print(f"\n🕐 Recent candles (last 48 hours): {len(recent_candles)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing data: {e}")
        return False

def main():
    print("🚀 SEP Trading System - Redis Market Data Seeder")
    print("=" * 50)
    
    # Check if we're running inside Docker
    if os.path.exists('/.dockerenv'):
        print("🐳 Running inside Docker container")
        # In Docker, Redis service is available as 'redis'
        os.environ.setdefault('REDIS_URL', 'redis://redis:6379/0')
    else:
        print("💻 Running on local machine")
        # Local development
        os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
    
    print(f"🔗 Redis URL: {os.environ.get('REDIS_URL')}")
    
    success = seed_redis_data()
    
    if success:
        print("\n🎉 Market data seeding completed successfully!")
        print("💡 You can now test the /api/market-data endpoint")
        print("🌐 Try: curl 'http://localhost:5000/api/market-data?instrument=EUR_USD'")
    else:
        print("\n❌ Market data seeding failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()