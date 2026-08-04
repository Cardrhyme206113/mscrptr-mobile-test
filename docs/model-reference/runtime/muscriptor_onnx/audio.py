"""NumPy implementation of MuScriptor's 16 kHz log-mel frontend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
N_FFT = 2_048
HOP_LENGTH = 160
N_MELS = 512
SEGMENT_SAMPLES = 5 * SAMPLE_RATE


def load_audio_16k(path: str | Path) -> np.ndarray:
    """Load an audio file as mono float32 at 16 kHz."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = np.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def first_five_second_chunk(audio: np.ndarray) -> np.ndarray:
    """Crop/pad audio to the five-second chunk used by MuScriptor."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= SEGMENT_SAMPLES:
        return audio[:SEGMENT_SAMPLES]
    return np.pad(audio, (0, SEGMENT_SAMPLES - audio.size))


def _hz_to_mel_htk(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz_htk(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """Match the pure-torch HTK filterbank bundled with MuScriptor."""
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1, dtype=np.float32)
    mel_min = _hz_to_mel_htk(np.asarray(0.0, dtype=np.float32))
    mel_max = _hz_to_mel_htk(np.asarray(SAMPLE_RATE / 2, dtype=np.float32))
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2, dtype=np.float32)
    freq_points = _mel_to_hz_htk(mel_points)
    freq_diff = freq_points[1:] - freq_points[:-1]
    slopes = freq_points[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / freq_diff[:-1]
    up = slopes[:, 2:] / freq_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Return `[1, 501, 512]` log-mel features for a five-second chunk.

    The reflection padding, periodic Hann window, FFT, HTK mel bank, and log
    epsilon mirror ``muscriptor.modules.mel_spectrogram``.
    """
    audio = first_five_second_chunk(audio)
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=-1)).astype(np.float32)
    mel = spectrum @ mel_filterbank()
    return np.log(mel + np.float32(1e-6))[None, ...].astype(np.float32)

