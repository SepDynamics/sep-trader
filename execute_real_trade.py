#!/usr/bin/env python3
"""
Real OANDA Trade Execution - Actually places trades on demo account
"""
import os
import requests
import json
import sys

def place_order(direction, units, stop_loss_pips=20, take_profit_pips=40, instrument="EUR_USD"):
    """Actually place a real trade on OANDA demo account"""
    
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    base_url = 'https://api-fxpractice.oanda.com'
    
    if not api_key or not account_id:
        print("ERROR: OANDA credentials not found")
        return False
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Convert pips to price distance (JPY pairs use 0.01, others use 0.0001)
    if 'JPY' in instrument:
        pip_value = 0.01  # JPY pairs
    else:
        pip_value = 0.0001  # Major pairs
    
    stop_distance = stop_loss_pips * pip_value
    profit_distance = take_profit_pips * pip_value
    
    # Generate unique client order ID for tracking
    import time
    client_order_id = f"SEP_{direction}_{int(time.time())}"
    client_trade_id = f"TRADE_{direction}_{int(time.time())}"
    
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": f"{'+' if direction == 'BUY' else '-'}{units}",
            "timeInForce": "FOK",  # Fill Or Kill - execute immediately or cancel
            "positionFill": "DEFAULT",  # Allow position netting
            "clientExtensions": {
                "id": client_order_id,
                "tag": "SEP_LIVE_TRADER",
                "comment": f"Live {direction} signal - 56.22% accuracy system"
            },
            "tradeClientExtensions": {
                "id": client_trade_id,
                "tag": "SEP_TRADE",
                "comment": f"Auto-trade {direction} based on quantum analysis"
            },
            "stopLossOnFill": {
                "distance": f"{stop_distance:.5f}",
                "timeInForce": "GTC",  # Good Till Cancelled
                "clientExtensions": {
                    "id": f"SL_{client_order_id}",
                    "tag": "SEP_STOP_LOSS"
                }
            },
            "takeProfitOnFill": {
                "distance": f"{profit_distance:.5f}",
                "timeInForce": "GTC",
                "clientExtensions": {
                    "id": f"TP_{client_order_id}",
                    "tag": "SEP_TAKE_PROFIT"
                }
            }
        }
    }
    
    print(f"🚀 PLACING REAL TRADE:")
    print(f"   Direction: {direction}")
    print(f"   Units: {units}")
    print(f"   Stop Loss: {stop_loss_pips} pips")
    print(f"   Take Profit: {take_profit_pips} pips")
    
    try:
        response = requests.post(
            f'{base_url}/v3/accounts/{account_id}/orders',
            headers=headers,
            data=json.dumps(order_data)
        )
        
        result = response.json()
        
        if response.status_code == 201:
            print("✅ TRADE EXECUTED SUCCESSFULLY!")
            
            # Extract all transaction details
            transactions = result.get('orderCreateTransaction', {})
            fill_transaction = result.get('orderFillTransaction', {})
            
            if transactions:
                print(f"   Order ID: {transactions.get('id', 'N/A')}")
                print(f"   Client Order ID: {transactions.get('clientExtensions', {}).get('id', 'N/A')}")
                print(f"   Time: {transactions.get('time', 'N/A')}")
            
            if fill_transaction:
                trade_id = fill_transaction.get('id', 'N/A')
                price = fill_transaction.get('price', 'N/A')
                units_filled = fill_transaction.get('units', 'N/A')
                financing = fill_transaction.get('financing', '0.0')
                commission = fill_transaction.get('commission', '0.0')
                
                print(f"   Trade ID: {trade_id}")
                print(f"   Fill Price: {price}")
                print(f"   Units Filled: {units_filled}")
                print(f"   Commission: ${commission}")
                print(f"   Financing: ${financing}")
                
                # Show stop loss and take profit orders created
                if 'stopLossOrderTransaction' in result:
                    sl_order = result['stopLossOrderTransaction']
                    print(f"   Stop Loss Order: {sl_order.get('id', 'N/A')} at distance {sl_order.get('distance', 'N/A')}")
                
                if 'takeProfitOrderTransaction' in result:
                    tp_order = result['takeProfitOrderTransaction']
                    print(f"   Take Profit Order: {tp_order.get('id', 'N/A')} at distance {tp_order.get('distance', 'N/A')}")
            
            return True
        else:
            print("❌ TRADE FAILED:")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {result}")
            return False
            
    except Exception as e:
        print(f"❌ TRADE ERROR: {e}")
        return False

def get_account_balance():
    """Get current account balance"""
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    base_url = 'https://api-fxpractice.oanda.com'
    
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        response = requests.get(f'{base_url}/v3/accounts/{account_id}', headers=headers)
        if response.status_code == 200:
            data = response.json()
            balance = float(data['account']['balance'])
            return balance
        return None
    except:
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 execute_real_trade.py BUY|SELL units [stop_pips] [profit_pips] [instrument]")
        sys.exit(1)
    
    direction = sys.argv[1]
    units = int(sys.argv[2])
    stop_pips = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    profit_pips = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    instrument = sys.argv[5] if len(sys.argv) > 5 else "EUR_USD"
    
    # Show current balance
    balance = get_account_balance()
    if balance:
        print(f"Current Balance: ${balance:.2f}")
    
    # Execute trade
    success = place_order(direction, units, stop_pips, profit_pips, instrument)
    
    if success:
        # Show new balance
        new_balance = get_account_balance()
        if new_balance:
            print(f"New Balance: ${new_balance:.2f}")
    
    sys.exit(0 if success else 1)
