import time
from utils.app_utils import get_binance_ohlcv

CACHE = {}
LAST_FETCH = {}
TTL = 10  # seconds (adjust based on how live you want it)

def get_cached_ohlcv(symbol: str):
    now = time.time()
    if symbol not in CACHE or now - LAST_FETCH[symbol] > TTL:
        CACHE[symbol] = get_binance_ohlcv(symbol)
        LAST_FETCH[symbol] = now
    return CACHE[symbol]
