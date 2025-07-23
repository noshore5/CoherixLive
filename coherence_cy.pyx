# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

def coherence_cy(np.ndarray[np.complex128_t, ndim=2] W1, np.ndarray[np.complex128_t, ndim=2] W2, np.ndarray[np.float64_t, ndim=1] freqs):
    cdef int n_freq = W1.shape[0]
    cdef int n_time = W1.shape[1]
    cdef np.ndarray[np.complex128_t, ndim=2] S1 = np.zeros((n_freq, n_time), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] S2 = np.zeros((n_freq, n_time), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] S12 = np.zeros((n_freq, n_time), dtype=np.complex128)
    cdef np.ndarray[np.float64_t, ndim=2] coh = np.zeros((n_freq, n_time), dtype=np.float64)
    cdef int i, j
    cdef int win = 6
    cdef double s1, s2
    cdef complex s12
    cdef int count, k
    for i in range(n_freq):
        for j in range(n_time):
            # windowed smoothing
            s1 = 0.0
            s2 = 0.0
            s12 = 0.0
            count = 0
            for k in range(max(0, j-win), min(n_time, j+win+1)):
                s1 += abs(W1[i, k])**2
                s2 += abs(W2[i, k])**2
                s12 += W1[i, k] * W2[i, k].conjugate()
                count += 1
            S1[i, j] = s1 / count
            S2[i, j] = s2 / count
            S12[i, j] = s12 / count
            if S1[i, j].real > 0 and S2[i, j].real > 0:
                coh[i, j] = abs(S12[i, j])**2 / (S1[i, j].real * S2[i, j].real)
            else:
                coh[i, j] = 0.0
    return coh, freqs, S12
