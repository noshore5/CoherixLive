
import numpy as np
from utils.app_utils import transform, coherence, get_binance_ohlcv


def plot_coherence(signal1, signal2, fs, highest=0.5, lowest=0.01):
    """
    Plots the wavelet coherence between two signals using the fcwt method.

    Args:
        signal1 (array-like): First input signal.
        signal2 (array-like): Second input signal.
        fs (float): Sampling frequency (Hz).
        highest (float): Highest frequency (Hz) for wavelet transform.
        lowest (float): Lowest frequency (Hz) for wavelet transform.
    """
    # Detrend signals
    from scipy.signal import detrend
    signal1 = detrend(np.asarray(signal1))
    signal2 = detrend(np.asarray(signal2))

    coeffs1, freqs = transform(signal1, fs, highest, lowest)
    coeffs2, _ = transform(signal2, fs, highest, lowest)
    # Always use the Python version for now (no Cython)
    coh, freqs, S12 = coherence(coeffs1, coeffs2, freqs)

    # Convert frequencies to periods (avoid division by zero)
    periods = np.where(freqs > 0, 1.0 / freqs, np.nan)
    X, Y = np.meshgrid(np.arange(signal1.shape[0]), periods)

    # No plotting here. Return data for frontend plotting.
    # NOTE: To ensure the coherence plot matches the detrended plot, call this function with the exact detrended arrays used in the frontend.
    return {
        'coherence': coh.tolist(),
        'freqs': freqs.tolist(),
        'periods': periods.tolist(),
        'phase': np.angle(S12).tolist()
    }

# Example usage (replace with your actual signals and sampling frequency):

'''
def plot_btc_eth_coherence_last_hour():
    """
    Fetches last hour of BTCUSDT and ETHUSDT close prices at 1s intervals from Binance,
    normalizes both to percentage change, and plots both the signals and their wavelet coherence.
    """
    # Fetch last hour (3600 seconds) of data
    btc = get_binance_ohlcv('BTCUSDT')[-3600:]
    eth = get_binance_ohlcv('ETHUSDT')[-3600:]
    if len(btc) < 10 or len(eth) < 10:
        print('Not enough data to plot coherence.')
        return
    btc_close = np.array([d['close'] for d in btc])
    eth_close = np.array([d['close'] for d in eth])

    # Normalize to percentage change from the first value
    btc_pct = 100 * (btc_close - btc_close[0]) / btc_close[0]
    eth_pct = 100 * (eth_close - eth_close[0]) / eth_close[0]
    fs = 1  # 1 Hz (1s interval)

    # No plotting here. Return data for frontend plotting.
    return {
        'btc_pct': btc_pct.tolist(),
        'eth_pct': eth_pct.tolist(),
        'coherence_result': plot_coherence(btc_pct, eth_pct, fs)
    }
'''