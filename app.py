from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import base64
import numpy as np

from utils.cache import get_cached_ohlcv
from plot_coherence import plot_coherence

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/ohlcv/{symbol}")
def api_ohlcv(symbol: str):
    data = get_cached_ohlcv(symbol.upper())
    return JSONResponse(content=data)

@app.get("/api/wavelet_coherence")
def api_wavelet_coherence():
    print("[API] /api/wavelet_coherence called")
    btc = get_cached_ohlcv('BTCUSDT')[-3600:]
    eth = get_cached_ohlcv('ETHUSDT')[-3600:]
    print(f"[API] BTCUSDT data points: {len(btc)}  ETHUSDT data points: {len(eth)}")
    if len(btc) < 10 or len(eth) < 10:
        print("[API] Not enough data for coherence computation")
        return JSONResponse(status_code=400, content={"error": "Not enough data"})
    btc_close = np.array([d['close'] for d in btc])
    eth_close = np.array([d['close'] for d in eth])
    print(f"[API] BTC close sample: {btc_close[:5]} ... ETH close sample: {eth_close[:5]} ...")
    try:
        coherence_result = plot_coherence(btc_close, eth_close, fs=1)
        print("[API] plot_coherence returned successfully")
        # Save result to static folder as JSON
        import json, os
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        static_json_path = os.path.join(static_dir, 'last_coherence.json')
        with open(static_json_path, 'w') as f:
            json.dump(coherence_result, f)
        print(f"[API] Coherence result saved to {static_json_path}")
        # Matplotlib PNG export removed. Chart.js PNG will be saved from frontend.
        coherence = np.array(coherence_result['coherence'])
        periods = np.array(coherence_result['periods'])
    except Exception as e:
        print(f"[API] plot_coherence failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(content=coherence_result)

# --- Endpoint to save Chart.js PNG from frontend ---
@app.post('/api/save_coherence_png')
async def save_coherence_png(request: Request):
    data = await request.json()
    img_b64 = data.get('image', '')
    if img_b64.startswith('data:image/png;base64,'):
        img_b64 = img_b64.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(img_b64)
        with open('static/last_coherence.png', 'wb') as f:
            f.write(img_bytes)
        return { 'status': 'ok' }
    except Exception as e:
        return { 'status': 'error', 'detail': str(e) }
