"""
Acoustic signal processing — Zone B.

VAD, pause detection, phonation segment extraction, and jitter/shimmer/HNR
extraction via parselmouth (Praat). Nonlinear dynamics (RPDE/DFA/PPE) come
from ml/nonlinear.py — see that module for why they aren't taken from nolds.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import webrtcvad
from scipy.io import wavfile

from ml.nonlinear import dfa as _dfa
from ml.nonlinear import ppe as _ppe
from ml.nonlinear import rpde as _rpde

logger = logging.getLogger(__name__)


@dataclass
class Pause:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def run_vad(audio_path: str, frame_duration_ms: int = 20) -> list[tuple[float, float]]:
    """Voice activity detection using webrtcvad.
    Returns list of (start_s, end_s) speech segments.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        frame_duration_ms: Frame length for VAD (10, 20, or 30)

    Returns:
        List of (start_s, end_s) tuples for detected speech segments
    """
    y, sr = librosa.load(audio_path, sr=16000)
    y_int16 = np.int16(y / np.max(np.abs(y)) * 32767)

    vad = webrtcvad.Vad()
    vad.set_mode(1)  # Aggressive mode (0-3, higher = more aggressive)

    frame_len = int(sr * frame_duration_ms / 1000)
    speech_segments = []
    segment_start = None

    for i in range(0, len(y_int16) - frame_len, frame_len):
        frame = y_int16[i : i + frame_len].tobytes()
        is_speech = vad.is_speech(frame, sr)

        if is_speech and segment_start is None:
            segment_start = i / sr
        elif not is_speech and segment_start is not None:
            speech_segments.append((segment_start, i / sr))
            segment_start = None

    if segment_start is not None:
        speech_segments.append((segment_start, len(y_int16) / sr))

    return speech_segments


def detect_pauses(audio_path: str, word_timestamps: list[dict]) -> list[Pause]:
    """Detect pauses from STT word-level timestamps (gaps between words).

    Args:
        audio_path: Path to audio file (not used, kept for API compatibility)
        word_timestamps: List of {"word": str, "start": float, "end": float}

    Returns:
        List of Pause objects for detected silence intervals
    """
    if not word_timestamps:
        return []

    pauses = []
    for i in range(len(word_timestamps) - 1):
        gap_start = word_timestamps[i]["end"]
        gap_end = word_timestamps[i + 1]["start"]
        gap_duration = gap_end - gap_start

        if gap_duration > 0.1:  # Minimum 100ms silence to count as pause
            pauses.append(Pause(start_s=gap_start, end_s=gap_end))

    return pauses


def extract_phonation_segment(audio_path: str, segment_duration: float = 3.0) -> str:
    """Extract a sustained phonation segment from audio.
    Uses silence detection to find continuous speech; returns longest coherent segment.

    Args:
        audio_path: Path to audio file
        segment_duration: Target duration for phonation segment (seconds)

    Returns:
        Path to temporary WAV file containing the extracted phonation segment
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # Use VAD to find speech segments
    speech_segments = run_vad(audio_path)

    if not speech_segments:
        raise ValueError(f"No speech detected in {audio_path}")

    # Find the longest continuous speech segment (likely the phonation task)
    longest_segment = max(speech_segments, key=lambda s: s[1] - s[0])
    start_sample = int(longest_segment[0] * sr)
    end_sample = int(min(longest_segment[1], longest_segment[0] + segment_duration) * sr)

    phonation_audio = y[start_sample:end_sample]

    # Save to temp file
    temp_file = Path(tempfile.gettempdir()) / f"phonation_{Path(audio_path).stem}.wav"
    wavfile.write(str(temp_file), sr, np.int16(phonation_audio * 32767))

    return str(temp_file)


def extract_acoustic_features(phonation_audio_path: str) -> dict:
    """Extract jitter/shimmer/HNR/RPDE/DFA/PPE from sustained phonation.

    Args:
        phonation_audio_path: Path to phonation segment audio file

    Returns:
        Dict with keys: jitter_local, jitter_rap, shimmer_local, shimmer_apq5,
                       hnr, rpde, dfa, ppe
    """
    sound = parselmouth.Sound(phonation_audio_path)

    # Pitch detection (required for jitter/shimmer and PPE).
    # NB: the kwargs are pitch_floor/pitch_ceiling — parselmouth has no
    # f0_min/f0_max parameters and raises TypeError if given them.
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=500)

    # Point process for jitter calculation
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # Jitter measurements
    jitter_local = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_rap = parselmouth.praat.call(
        point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )

    # Shimmer measurements. Praat's shimmer commands need BOTH the sound and
    # the point process — passing the sound alone raises
    # "Command not available for given objects".
    shimmer_local = parselmouth.praat.call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq5 = parselmouth.praat.call(
        [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # HNR (Harmonic-to-Noise Ratio). "Get mean" on a Harmonicity object takes
    # a time range (0, 0 = whole file); without it Praat rejects the command.
    harmonicity = sound.to_harmonicity_cc()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    # Nonlinear dynamics features (see ml/nonlinear.py).
    # Failures are logged rather than silently swallowed: a feature that
    # quietly becomes 0.0 for every call feeds a constant into the UPDRS
    # model, which is worse than a visible error.
    y, sr = librosa.load(phonation_audio_path, sr=16000)

    try:
        rpde = _rpde(y, emb_dim=4, delay=int(0.001 * sr) or 1)
    except Exception:
        logger.warning("RPDE extraction failed for %s; using 0.0", phonation_audio_path, exc_info=True)
        rpde = 0.0

    try:
        dfa = _dfa(y)
    except Exception:
        logger.warning("DFA extraction failed for %s; using 0.0", phonation_audio_path, exc_info=True)
        dfa = 0.0

    try:
        # NB: selected_array is a structured array property, not a method —
        # `pitch.selected_array("frequency")` raises "not callable".
        pitch_values = pitch.selected_array["frequency"]
        ppe = _ppe(pitch_values)
    except Exception:
        logger.warning("PPE extraction failed for %s; using 0.0", phonation_audio_path, exc_info=True)
        ppe = 0.0

    return {
        "jitter_local": float(jitter_local),
        "jitter_rap": float(jitter_rap),
        "shimmer_local": float(shimmer_local),
        "shimmer_apq5": float(shimmer_apq5),
        "hnr": float(hnr),
        "rpde": float(rpde),
        "dfa": float(dfa),
        "ppe": float(ppe),
    }


def extract_prosody_features(audio_path: str, word_timestamps: list[dict]) -> dict:
    """Extract speech rate and pause statistics.

    Args:
        audio_path: Path to audio file
        word_timestamps: List of {"word": str, "start": float, "end": float}

    Returns:
        Dict with keys: speech_rate, pause_freq, pause_avg_duration
    """
    if not word_timestamps:
        return {
            "speech_rate": 0.0,
            "pause_freq": 0.0,
            "pause_avg_duration": 0.0,
        }

    # Get audio duration
    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(y) / sr

    # Calculate speech rate (words per minute)
    total_speech_time = sum(w["end"] - w["start"] for w in word_timestamps)
    num_words = len(word_timestamps)
    speech_rate = (num_words / total_speech_time * 60) if total_speech_time > 0 else 0.0

    # Detect pauses
    pauses = detect_pauses(audio_path, word_timestamps)
    pause_freq = len(pauses) / total_duration if total_duration > 0 else 0.0
    pause_avg_duration = (
        sum(p.duration_s for p in pauses) / len(pauses) if pauses else 0.0
    )

    return {
        "speech_rate": float(speech_rate),
        "pause_freq": float(pause_freq),
        "pause_avg_duration": float(pause_avg_duration),
    }
