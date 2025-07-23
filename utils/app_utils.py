from binance.client import Client
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend 
import fcwt  # Ensure you have the fcwt package installed
import numpy as np
import os

def get_binance_client():
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    return Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_binance_ohlcv(symbol):
    client = get_binance_client()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=3)
    klines = client.get_historical_klines(
        symbol.upper(),
        Client.KLINE_INTERVAL_1SECOND,
        start_time.strftime('%Y-%m-%d %H:%M:%S'),
        end_time.strftime('%Y-%m-%d %H:%M:%S')
    )
    data = [
        {
            # Binance returns timestamp in ms, but sometimes as int or str. Ensure int.
            'timestamp': int(k[0]),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        }
        for k in klines
    ]
    return data

def coherence(coeffs1, coeffs2, freqs):
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(1, 1), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smooth = smooth(S1)
    S2_smooth = smooth(S2)
    S12_smooth = smooth(np.abs(S12) ** 2)
    coh = S12_smooth / (np.sqrt(S1_smooth) * np.sqrt(S2_smooth))
    return coh, freqs, S12

def transform(signal1, frame_rate, highest, lowest):
    nfreqs = 100
    freqs, coeffs1 = fcwt.cwt(signal1, frame_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
    return coeffs1, freqs
