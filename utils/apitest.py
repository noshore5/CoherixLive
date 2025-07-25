import requests

API_KEY = "d21mdr9r01qpst75oaa0d21mdr9r01qpst75oaag"  # Replace with your real key
url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={API_KEY}"

import websocket
import json
import time
from datetime import datetime
from collections import deque

# Your Finnhub API Key
FINNHUB_API_KEY = API_KEY  # Use the correct API key variable
SYMBOL = "AAPL"

# Rolling window: 1800 seconds (30 minutes)
price_window = deque([0.0]*1800, maxlen=1800)
last_price = 0.0
last_timestamp = None

def on_message(ws, message):
    global last_price, last_timestamp

    data = json.loads(message)

    if 'data' in data:
        for item in data['data']:
            price = item['p']
            timestamp = item['t'] // 1000  # Convert from ms to seconds
            now = int(time.time())

            # Fill missing seconds if necessary
            if last_timestamp is not None and timestamp > last_timestamp + 1:
                for _ in range(timestamp - last_timestamp - 1):
                    price_window.append(last_price)

            price_window.append(price)
            last_price = price
            last_timestamp = timestamp

            print(f"{datetime.fromtimestamp(timestamp)} | Price: {price} | Length: {len(price_window)}")

def on_open(ws):
    print("WebSocket connection opened")
    auth = {"type": "auth", "token": FINNHUB_API_KEY}
    ws.send(json.dumps(auth))
    sub = {"type": "subscribe", "symbol": SYMBOL}
    ws.send(json.dumps(sub))

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_error(ws, error):
    print("WebSocket error:", error)

if __name__ == "__main__":
    # Make sure you have installed the correct websocket-client package:
    # pip install websocket-client
    # NOT the 'websocket' package!
    socket = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    ws = websocket.WebSocketApp(
        socket,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error
    )
    ws.run_forever()
