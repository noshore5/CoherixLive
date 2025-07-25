from scipy.ndimage import gaussian_filter
from scipy.signal import detrend 
import fcwt  # Ensure you have the fcwt package installed
import numpy as np
import os

def coherence(coeffs1, coeffs2, freqs):
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(.5, .5), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smooth = smooth(S1)
    S2_smooth = smooth(S2)
    S12_smooth = smooth(np.abs(S12) ** 2)
    coh = S12_smooth / (np.sqrt(S1_smooth) * np.sqrt(S2_smooth))
    return coh, freqs, S12

def transform(signal1, frame_rate, highest, lowest, nfreqs=100):
    """Transform signal using continuous wavelet transform.
    Parameters are scaled based on frame_rate."""
    signal1 = np.asarray(signal1, dtype=np.float64)
    
    # If sampling rate is high (60s tab), use higher frequency range
    if frame_rate >= 30:
        # For 30Hz sampling, use periods 1-15 seconds
        lowest = float(1/15.0)  # longest period = 15 seconds
        highest = float(2)    # shortest period = 1 second
    else:
        print(f'[DEBUG] Frame rate: {frame_rate}')
        # For 1Hz sampling (30m tab), use periods 40-500 seconds
        lowest = .002  # longest period = 500 seconds
        highest = .025 # shortest period = 40 seconds
        frame_rate = int(frame_rate)  # Ensure frame_rate is an integer for fcwt compatibility

    freqs, coeffs1 = fcwt.cwt(signal1, frame_rate, lowest, highest, nfreqs, 
                             nthreads=4, scaling='log')
    print(f'[DEBUG] Transforming signal with frame_rate={frame_rate}')
    return coeffs1, freqs
