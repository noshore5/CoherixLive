from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from utils.app_utils import get_binance_ohlcv
from plot_coherence import plot_coherence
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/ohlcv/{symbol}")
async def get_ohlcv(symbol: str):
    data = get_binance_ohlcv(symbol)
    return JSONResponse(content=data)

@app.get("/api/wavelet_coherence")
async def get_wavelet_coherence():
    btc = get_binance_ohlcv('BTCUSDT')[-3600:]
    eth = get_binance_ohlcv('ETHUSDT')[-3600:]
    if len(btc) < 10 or len(eth) < 10:
        raise HTTPException(status_code=400, detail="Not enough data")
    btc_close = np.array([d['close'] for d in btc])
    eth_close = np.array([d['close'] for d in eth])
    data = plot_coherence(btc_close, eth_close, fs=1)
    return JSONResponse(content=data)
