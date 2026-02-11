# analyzer_core_adaptive_frequency_core
```python
# analyzercore_adaptive_frequency_core.py
# ============================================================================
# Adaptive Frequency Analysis Core
#
# Provides:
#   - Zero-Crossing Rate (ZCR)
#   - Adaptive Pitch Tracking
#   - Adaptive Time-Domain Spectral Centroid
#
# Time-domain, continuous, FFT-free analyzers inspired by Faust analyzers.lib
# ============================================================================
from __future__ import annotations
import math
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def d(f): return f
        return d

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception:
    _HAVE_SD = False


# ============================================================================
# Utilities (DSP-safe)
# ============================================================================

@njit(cache=True, fastmath=True)
def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True, fastmath=True)
def _alpha(sr, tau):
    tau = _clamp(tau, 1e-4, 10.0)
    return math.exp(-1.0 / (tau * sr))


# ============================================================================
# INIT / UPDATE
# ============================================================================

def adaptive_frequency_core_init(sr: float, eps: float = 1e-12) -> tuple:
    sr = max(8000.0, float(sr))
    eps = max(1e-20, eps)

    state = (
        sr, eps,
        0.0,    # prev_x
        0.0,    # zcr_lp
        0.0,    # lpf_state
        100.0,  # pitch_est
        0.01,   # centroid_fc (normalized)
        0.0,    # rms_total
        0.0,    # rms_low
        0.0,    # rms_high
    )
    return state


def adaptive_frequency_core_update_state(state: tuple) -> tuple:
    return state


# ============================================================================
# TICK DSP CORE
# ============================================================================

@njit(cache=True, fastmath=True)
def adaptive_frequency_core_tick(
    state: tuple,
    x: float,
    tau: float,
    nonlinearity: int,
) -> tuple:
    sr, eps, prev_x, zcr_lp, lpf_state, pitch_est, fc_norm, rms_t, rms_l, rms_h = state

    tau = _clamp(tau, 1e-4, 1.0)
    a = _alpha(sr, tau)

    # ---------------- ZCR ----------------
    zc = 1.0 if (x >= 0.0) != (prev_x >= 0.0) else 0.0
    zcr_lp = (1.0 - a) * zc + a * zcr_lp
    zcr = zcr_lp

    # ---------------- Pitch ----------------
    pitch_est = _clamp(zcr * sr * 0.5, 20.0, sr * 0.5)

    # ---------------- Adaptive LPF ----------------
    wc = 2.0 * math.pi * pitch_est / sr
    g = wc / (1.0 + wc)
    lpf_state = lpf_state + g * (x - lpf_state)

    # ---------------- Spectral Centroid ----------------
    low = lpf_state
    high = x - low

    rms_t = (1.0 - a) * (x * x) + a * rms_t
    rms_l = (1.0 - a) * (low * low) + a * rms_l
    rms_h = (1.0 - a) * (high * high) + a * rms_h

    diff = (rms_h - rms_l) / max(rms_t, eps)
    if nonlinearity:
        diff = diff * diff * diff

    fc_norm = _clamp(fc_norm + diff * 0.001, 0.0, 1.0)
    centroid = _clamp(fc_norm * sr * 0.5, 20.0, sr * 0.5)

    new_state = (
        sr, eps,
        x,
        zcr_lp,
        lpf_state,
        pitch_est,
        fc_norm,
        rms_t,
        rms_l,
        rms_h,
    )
    return (zcr, pitch_est, centroid), new_state


# ============================================================================
# BLOCK DSP CORE
# ============================================================================

@njit(cache=True, fastmath=True)
def adaptive_frequency_core_process_block(
    state: tuple,
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    nonlinearity: int,
) -> tuple:
    st = state
    for i in range(x.shape[0]):
        out, st = adaptive_frequency_core_tick(st, x[i], tau, nonlinearity)
        y[i, 0] = out[0]
        y[i, 1] = out[1]
        y[i, 2] = out[2]
    return y, st


# ============================================================================
# WRAPPER (non-DSP)
# ============================================================================

class AdaptiveFrequencyCore:
    def __init__(self, sr=48000):
        self.state = adaptive_frequency_core_init(sr)
        self.tau = 0.02
        self.nl = 1

    def tick(self, x):
        y, self.state = adaptive_frequency_core_tick(self.state, x, self.tau, self.nl)
        return y

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros((x.shape[0], 3), dtype=np.float64)
        y, self.state = adaptive_frequency_core_process_block(
            self.state, x, y, self.tau, self.nl
        )
        return y


# ============================================================================
# TESTS / PLOTS / LISTENING
# ============================================================================

if __name__ == "__main__":
    sr = 48000
    t = np.arange(int(2.0 * sr)) / sr

    # Sweep test
    f = 50.0 * (4000.0 / 50.0) ** (t / t[-1])
    x = 0.6 * np.sin(2 * np.pi * f * t)

    core = AdaptiveFrequencyCore(sr)
    y = core.process(x)

    if _HAVE_MPL:
        plt.figure(figsize=(14, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, x)
        plt.title("Input Signal")

        plt.subplot(3, 1, 2)
        plt.plot(t, y[:, 1])
        plt.title("Pitch Estimate (Hz)")

        plt.subplot(3, 1, 3)
        plt.plot(t, y[:, 2])
        plt.title("Adaptive Spectral Centroid (Hz)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    if _HAVE_SD:
        print("Listening test...")
        sd.play(x, sr)
        sd.wait()
```
