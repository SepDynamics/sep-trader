#!/usr/bin/env python3
"""
Test script for OANDA API integration and candle data functionality.
This script tests:
1. OANDA API connectivity
2. Candle data retrieval
3. Database storage operations
4. Trading service API endpoints
"""

import os
import sys
import json
import requests
import time
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_connection import ValkeyConnection
from oanda_connector import OandaConnector

def test_oanda_connectivity():
    """Test OANDA API connectivity"""
    print("=" * 60)
    print("1. TESTING OANDA API CONNECTIVITY")
    print("=" * 60)
    
    try:
        oanda = OandaConnector()
        
        # Test account information
        print("Testing account information...")
        account_info = oanda.get_account_info()
        if account_info:
            print(f"✅ Account connected: {account_info.get('id', 'Unknown')}")
            print(f"   Currency: {account_info.get('currency', 'Unknown')}")
            print(f"   Balance: {account_info.get('balance', 'Unknown')}")
        else:
            print("❌ Failed to retrieve account information")
            return False
        
        # Test instruments listing
        print("Testing instruments listing...")
        instruments = oanda.get_instruments()
        if instruments:
            print(f"✅ Retrieved {len(instruments)} instruments")
            print(f"   Sample instruments: {[inst.get('name') for inst in instruments[:5]]}")
        else:
            print("❌ Failed to retrieve instruments")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ OANDA connectivity test failed: {e}")
        return False

def test_candle_data_retrieval():
    """Test candle data retrieval from OANDA"""
    print("\n" + "=" * 60)
    print("2. TESTING CANDLE DATA RETRIEVAL")
    print("=" * 60)
    
    try:
        oanda = OandaConnector()
        
        # Test instruments to fetch candle data for
        test_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        
        for instrument in test_instruments:
            print(f"\nTesting candle data for {instrument}...")
            
            # Test latest candles
            candles = oanda.get_latest_candles(instrument, 'M5', 10)
            if candles:
                print(f"✅ Retrieved {len(candles)} candles for {instrument}")
                
                # Show sample candle data
                if candles:
                    sample = candles[0]
                    print(f"   Sample candle: Time={sample.get('time', 'Unknown')}")
                    print(f"                  Open={sample.get('mid', {}).get('o', 'Unknown')}")
                    print(f"                  High={sample.get('mid', {}).get('h', 'Unknown')}")
                    print(f"                  Low={sample.get('mid', {}).get('l', 'Unknown')}")
                    print(f"                  Close={sample.get('mid', {}).get('c', 'Unknown')}")
            else:
                print(f"❌ Failed to retrieve candles for {instrument}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Candle data retrieval test failed: {e}")
        return False

def test_database_storage():
    """Test database storage functionality"""
    print("\n" + "=" * 60)
    print("3. TESTING DATABASE STORAGE")
    print("=" * 60)
    
    try:
        # Test database connection
        print("Testing database connection...")
        db = ValkeyConnection()
        
        health = db.health_check()
        if not health:
            print("❌ Database health check failed")
            return False
        
        print("✅ Database connection healthy")
        
        # Test candle data storage and retrieval
        print("\nTesting candle data storage...")
        
        # Create sample candle data
        sample_candles = [
            {
                'time': '2024-01-01T12:00:00.000000000Z',
                'volume': 1000,
                'complete': True,
                'mid': {
                    'o': '1.1000',
                    'h': '1.1050',
                    'l': '1.0950',
                    'c': '1.1025'
                }
            },
            {
                'time': '2024-01-01T12:05:00.000000000Z',
                'volume': 1200,
                'complete': True,
                'mid': {
                    'o': '1.1025',
                    'h': '1.1075',
                    'l': '1.0975',
                    'c': '1.1050'
                }
            }
        ]
        
        # Test storage
        instrument = 'TEST_PAIR'
        granularity = 'M5'
        
        success = db.store_candle_data(instrument, granularity, sample_candles)
        if success:
            print(f"✅ Successfully stored {len(sample_candles)} test candles")
        else:
            print("❌ Failed to store test candles")
            return False
        
        # Test retrieval
        print("Testing candle data retrieval...")
        retrieved_candles = db.get_candle_data(instrument, granularity, 10)
        
        if retrieved_candles and len(retrieved_candles) >= len(sample_candles):
            print(f"✅ Successfully retrieved {len(retrieved_candles)} candles")
        else:
            print(f"❌ Failed to retrieve candles (expected >= {len(sample_candles)}, got {len(retrieved_candles) if retrieved_candles else 0})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database storage test failed: {e}")
        return False

def test_trading_service_api():
    """Test trading service API endpoints"""
    print("\n" + "=" * 60)
    print("4. TESTING TRADING SERVICE API")
    print("=" * 60)
    
    base_url = "http://localhost:8080"
    
    try:
        # Test health endpoint
        print("Testing /health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint responding")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test manual candle fetch endpoint
        print("\nTesting manual candle fetch...")
        fetch_data = {
            "instrument": "EUR_USD",
            "granularity": "M5",
            "count": 10
        }
        
        response = requests.post(
            f"{base_url}/api/candles/fetch",
            json=fetch_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Manual candle fetch successful")
            else:
                print("❌ Manual candle fetch reported failure")
                return False
        else:
            print(f"❌ Manual candle fetch failed: {response.status_code}")
            return False
        
        # Test candle data retrieval endpoint
        print("\nTesting candle data retrieval endpoint...")
        time.sleep(2)  # Give some time for data to be stored
        
        response = requests.get(
            f"{base_url}/api/candles/EUR_USD?granularity=M5&limit=10",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            candles = result.get('candles', [])
            if len(candles) > 0:
                print(f"✅ Retrieved {len(candles)} candles via API")
            else:
                print("❌ No candles returned from API")
                return False
        else:
            print(f"❌ Candle retrieval endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to trading service (is it running on port 8080?)")
        return False
    except Exception as e:
        print(f"❌ Trading service API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 OANDA INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    tests = [
        ("OANDA Connectivity", test_oanda_connectivity),
        ("Candle Data Retrieval", test_candle_data_retrieval),
        ("Database Storage", test_database_storage),
        ("Trading Service API", test_trading_service_api)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)