from fastapi import FastAPI, Request, Query, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, conlist
from typing import List
import numpy as np
import uvicorn
from scipy.signal import detrend
import aiohttp
import asyncio
from datetime import datetime, timedelta
import random

from utils.coherence_utils import coherence, transform
from utils.finnhub_utils import get_historical_data, start_live_price_updates, get_live_price_array

app = FastAPI()

# Static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# HTML route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("coherixlive.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    # Start live price updates for AAPL and MSFT at app startup
    start_live_price_updates(['AAPL', 'MSFT'])

# State variables for price arrays
_price_arrays_initialized = False
_aapl_prices = None
_msft_prices = None
_random_state = random.Random(42)  # Fixed seed for reproducibility
_volatility = 0.001  # Small volatility for realistic price movements
_last_update_time = None  # Track when we last updated prices

# State variables for both tabs
_price_means = {'AAPL': 100.0, 'MSFT': 100.0}
_base_volatility = 0.0003
_mean_reversion = 0.05

def initialize_price_arrays(length=1800, start_aapl=150.0, start_msft=300.0):
    """Initialize price arrays with fixed starting values and random walk."""
    global _aapl_prices, _msft_prices, _price_arrays_initialized
    
    # Initialize AAPL prices
    aapl = [start_aapl]
    for _ in range(length - 1):
        change = _random_state.gauss(0, _volatility)
        new_price = aapl[-1] * (1 + change)
        aapl.append(max(0.01, new_price))
    
    # Initialize MSFT prices with same pattern but different base
    msft = [start_msft]
    for _ in range(length - 1):
        change = _random_state.gauss(0, _volatility)
        new_price = msft[-1] * (1 + change)
        msft.append(max(0.01, new_price))
    
    _aapl_prices = aapl
    _msft_prices = msft
    _price_arrays_initialized = True
    print(f"[DEBUG] Initialized price arrays - AAPL: {_aapl_prices[0]:.2f} to {_aapl_prices[-1]:.2f}, MSFT: {_msft_prices[0]:.2f} to {_msft_prices[-1]:.2f}")

def update_price_arrays(force_update=False):
    global _aapl_prices, _msft_prices, _last_update_time
    
    current_time = datetime.now()
    if _last_update_time is not None and not force_update:
        time_since_update = (current_time - _last_update_time).total_seconds()
        if time_since_update < 5:
            return False

    # Simply rotate the arrays
    _aapl_prices = _aapl_prices[1:] + [_aapl_prices[0]]
    _msft_prices = _msft_prices[1:] + [_msft_prices[0]]
    
    _last_update_time = current_time
    return True

def get_pct_change_series(prices):
    base = prices[0] if prices and prices[0] != 0 else 1.0
    return [100 * (p - base) / base for p in prices]

@app.get("/api/live-data")
async def live_data():
    global _price_arrays_initialized, _aapl_prices, _msft_prices
    
    # Initialize arrays if needed
    if not _price_arrays_initialized:
        initialize_price_arrays()
        _price_arrays_initialized = True

    # Update the arrays (cycle through the pattern)
    update_price_arrays()
    
    # Create time axis
    now = datetime.now()
    time_axis = [(now - timedelta(seconds=1799-i)).strftime("%H:%M:%S") for i in range(1800)]
    
    return {
        "time": time_axis,
        "aapl": _aapl_prices,
        "msft": _msft_prices
    }

class DetrendRequest(BaseModel):
    djia: conlist(float, min_length=2)
    nasdaq: conlist(float, min_length=2)

@app.post("/api/detrend")
async def detrend_data(data: DetrendRequest):
    try:
        # Get arrays and ensure they exist
        aapl = data.djia or data.aapl or []
        msft = data.nasdaq or data.msft or []
        
        if len(aapl) == 0 or len(msft) == 0:
            return {
                "aapl_detrended": [],
                "msft_detrended": []
            }
            
        if len(aapl) != len(msft):
            raise HTTPException(
                status_code=422,
                detail="Arrays must have same length"
            )
        
        # The lag is likely caused by using scipy.signal.detrend on very large arrays (length 1800) every 2 seconds.
        # If you want to speed this up, you can use numpy's polyfit for a linear detrend, which is much faster:
        def fast_detrend(arr):
            x = np.arange(len(arr))
            p = np.polyfit(x, arr, 1)
            trend = np.polyval(p, x)
            return (arr - trend).tolist()

        aapl_arr = np.array(aapl, dtype=np.float64)
        msft_arr = np.array(msft, dtype=np.float64)
        aapl_detrended = fast_detrend(aapl_arr)
        msft_detrended = fast_detrend(msft_arr)
        
        return {
            "aapl_detrended": aapl_detrended,
            "msft_detrended": msft_detrended
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/coherence")
async def calculate_coherence(data: dict = Body(...)):
    try:
        aapl = data.get("aapl") or data.get("djia", [])
        msft = data.get("msft") or data.get("nasdaq", [])
        frame_rate = data.get("frame_rate", 1.0)
        
        if frame_rate == 30:
            highest_freq = frame_rate/2  # Nyquist frequency
            lowest_freq = 1/15   # 15 second period
            nfreqs = 200  # Higher resolution for high-frequency analysis
        else:
            highest_freq = 1/40
            lowest_freq = 1/500
            nfreqs = 200

        if not isinstance(aapl, list) or not isinstance(msft, list):
            raise HTTPException(status_code=422, detail="Data must be lists")
        if len(aapl) < 2 or len(msft) < 2:
            raise HTTPException(status_code=422, detail="Insufficient data points")
        
        aapl_arr = np.array([float(x) for x in aapl], dtype=np.float64)
        msft_arr = np.array([float(x) for x in msft], dtype=np.float64)
        
        # Handle arrays of all zeros
        if not np.any(aapl_arr) or not np.any(msft_arr):
            return {
                "coherence": [[0.0]],
                "periods": [1.0],
                "time": [0],
                "phase": [[0.0]]
            }
        
        print(f'[DEBUG] Frame rate: {frame_rate}, Highest freq: {highest_freq}, Lowest freq: {lowest_freq}')
        coeffs1, freqs = transform(aapl_arr, frame_rate, highest_freq, lowest_freq, nfreqs=nfreqs)
        coeffs2, _ = transform(msft_arr, frame_rate, highest_freq, lowest_freq, nfreqs=nfreqs)
        coh, freqs, cross = coherence(coeffs1, coeffs2, freqs)
        periods = 1 / freqs

        # Replace NaN values with zeros
        coh = np.nan_to_num(coh, nan=0.0)
        cross = np.nan_to_num(cross, nan=0.0)
        
        # Normalize coherence array
        coh = np.abs(coh)
        max_coh = np.max(coh)
        if max_coh > 0:
            coh = coh / max_coh

        return {
            "coherence": coh.tolist(),
            "periods": periods.tolist(),
            "time": list(range(coh.shape[1])),
            "phase": np.angle(cross).tolist()
        }
    except Exception as e:
        print(f"Coherence error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/finnhub")
async def websocket_finnhub(websocket: WebSocket):
    await websocket.accept()
    async for data in connect_finnhub_websocket():
        await websocket.send_json(data)

# State for high-frequency Markov chains
_hf_price_means = {'A': 100.0, 'B': 100.0}
_hf_prices_a = None
_hf_prices_b = None
_hf_volatility = 0.0003
_hf_mean_reversion = 0.05

@app.get("/api/hf-data")
async def get_hf_data():
    global _hf_prices_a, _hf_prices_b
    
    # Initialize if needed
    if _hf_prices_a is None or _hf_prices_b is None:
        _hf_prices_a = [_hf_price_means['A']] * 1800
        _hf_prices_b = [_hf_price_means['B']] * 1800

    # Generate new prices with mean reversion
    def update_prices(prices, mean):
        prices = prices[30:]  # Remove oldest 30 points
        current = prices[-1]
        new_prices = []
        for _ in range(30):
            mean_force = _hf_mean_reversion * (mean - current)
            random_walk = random.gauss(0, _hf_volatility)
            jump = random.uniform(-0.05, 0.05)
            current = current * (1 + mean_force + random_walk + jump)
            current = max(0.01, current)
            new_prices.append(current)
        return prices + new_prices

    # Update both price series
    _hf_prices_a = update_prices(_hf_prices_a, _hf_price_means['A'])
    _hf_prices_b = update_prices(_hf_prices_b, _hf_price_means['B'])

    # Convert to percent change
    base_a = _hf_prices_a[0]
    base_b = _hf_prices_b[0]
    pct_a = [(p - base_a) / base_a * 100 for p in _hf_prices_a]
    pct_b = [(p - base_b) / base_b * 100 for p in _hf_prices_b]

    # Generate time axis
    now = datetime.now()
    time_axis = [(now - timedelta(seconds=1799-i)).strftime("%H:%M:%S") for i in range(1800)]

    return {
        "time": time_axis,
        "aapl": pct_a,
        "msft": pct_b
    }

def generate_markov_chain(length=1800, start=100.0, prev_series=None, volatility=0.001, seed=None):
    """
    Generate or extend a synthetic price series using a simple Markov process.
    To avoid cycles, use a random seed that changes over time or add a small random drift.
    """
    # Use a random seed that changes (e.g., with time or a global counter)
    # Use a random seed that changes (e.g., with time or a global counter)
    rand = random.Random(seed if seed is not None else random.SystemRandom().randint(0, 1 << 30))
    drift = rand.gauss(0, volatility / 10)  # Small drift to avoid cycles

    if prev_series is not None and len(prev_series) == length:
        prices = prev_series[:]
        last_valid = next((p for p in reversed(prices[:-1]) if p != 0), start)
        change = rand.gauss(0, volatility) + drift
        new_price = max(0.01, last_valid * (1 + change))

        prices[-1] = new_price
    else:
        prices = [start]
        for _ in range(length - 1):
            change = rand.gauss(0, volatility) + drift
            new_price = max(0.01, prices[-1] * (1 + change))
            prices.append(new_price)
    # ...existing code for base and pct_changes if needed...
    return prices

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
