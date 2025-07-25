import finnhub
import numpy as np
from datetime import datetime, timedelta
import asyncio
import websockets
import json
import os
import websocket
import threading
from collections import deque

FINNHUB_KEY = os.getenv('FINNHUB_KEY', 'd21mdr9r01qpst75oaa0d21mdr9r01qpst75oaag')
client = finnhub.Client(api_key=FINNHUB_KEY)

async def connect_finnhub_websocket():
    uri = f"wss://ws.finnhub.io?token={FINNHUB_KEY}"
    
    async with websockets.connect(uri) as ws:
        # Subscribe to BTC and ETH (crypto symbols)
        await ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')
        await ws.send('{"type":"subscribe","symbol":"BINANCE:ETHUSDT"}')
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                if data['type'] == 'trade':
                    yield data
            except Exception as e:
                print(f"WebSocket error: {e}")
                break

def get_historical_data(symbol, minutes=30):
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        candles = client.stock_candles(
            symbol,
            'D',  # Use daily resolution
            start_ts,
            end_ts
        )
        if candles['s'] != 'ok':
            return {"time": [], "close": []}
        closes = np.array(candles['c'])
        if len(closes) > 0:
            closes = 100 * (closes - closes[0]) / closes[0]
        return {
            'time': candles['t'],
            'close': closes.tolist()
        }
    except Exception as e:
        # Return empty arrays on error (e.g., 403 Forbidden)
        return {"time": [], "close": []}

# Add a function to yield live prices from the websocket for a given symbol
def start_finnhub_websocket(symbols, on_price):
    FINNHUB_KEY = os.getenv('FINNHUB_KEY', 'd21mdr9r01qpst75oaa0d21mdr9r01qpst75oaag')
    socket = f"wss://ws.finnhub.io?token={FINNHUB_KEY}"
    print(f"[DEBUG] Starting Finnhub WebSocket connection for symbols: {symbols}")

    def on_message(ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            print(f"[DEBUG] Received message type: {msg_type}")
            
            if msg_type == 'ping':
                pong = json.dumps({"type": "pong"})
                ws.send(pong)
                return
                
            elif msg_type == 'trade':
                print(f"[DEBUG] Full trade data: {data}")
                for item in data.get('data', []):
                    symbol = item.get('s', '')
                    if symbol in symbols:  # Only process subscribed symbols
                        price = item.get('p', 0.0)
                        timestamp = item.get('t', 0) // 1000
                        print(f"[DEBUG] Valid trade: {symbol} @ {price}")
                        on_price(symbol, price, timestamp)
                        
            elif msg_type == 'error':
                print(f"[ERROR] Finnhub error: {data}")
                
        except Exception as e:
            print(f"[ERROR] Message processing error: {str(e)}")

    def on_open(ws):
        print("[DEBUG] WebSocket connection opened")
        ws.send(json.dumps({"type": "auth", "token": FINNHUB_KEY}))
        # Subscribe to crypto tickers
        for symbol in symbols:
            sub_msg = json.dumps({
                "type": "subscribe",
                "symbol": symbol
            })
            print(f"[DEBUG] Subscribing to: {sub_msg}")
            ws.send(sub_msg)

    def on_error(ws, error):
        print(f"[ERROR] WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"[DEBUG] WebSocket closed: {close_status_code} - {close_msg}")

    ws = websocket.WebSocketApp(
        socket,
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )

    thread = threading.Thread(target=ws.run_forever, daemon=True)
    thread.start()
    print("[DEBUG] WebSocket thread started")
    return ws  # You can keep a reference to close it later if needed

# In-memory rolling price arrays for each symbol
_price_windows = {}
_base_prices = {}
_window_length = 1800
_got_first_price = {}  # Track if we've received the first price

def _init_price_window(symbol):
    if symbol not in _price_windows:
        _price_windows[symbol] = deque([None]*_window_length, maxlen=_window_length)
        _base_prices[symbol] = None
        _got_first_price[symbol] = False

def handle_price(symbol, price, timestamp):
    _init_price_window(symbol)
    
    # Set base price on first received price
    if not _got_first_price[symbol]:
        _base_prices[symbol] = price
        # Fill entire window with 0.0 (no change from base price)
        _price_windows[symbol] = deque([0.0]*_window_length, maxlen=_window_length)
        _got_first_price[symbol] = True
        print(f"[DEBUG] Initialized {symbol} with base price: {price}")
    
    # Calculate percent change from base price
    if _base_prices[symbol] is not None:
        pct_change = 100 * (price - _base_prices[symbol]) / _base_prices[symbol]
        _price_windows[symbol].append(pct_change)
        print(f"[DEBUG] New price for {symbol}: {price}, %change: {pct_change:.2f}%")

def start_live_price_updates(symbols=['AAPL', 'MSFT']):
    # Start the websocket and fill the rolling windows
    start_finnhub_websocket(symbols, handle_price)

def get_live_price_array(symbol):
    _init_price_window(symbol)
    return list(_price_windows[symbol])

# Example usage:
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
    if not _got_first_price.get(key, False):
        print(f"[DEBUG] No prices received yet for {key}")
        return [0.0] * _window_length
    return list(_price_windows[key])

# Example usage:
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
# start_live_price_updates(['AAPL', 'MSFT'])
# print(get_live_price_array('AAPL'))
