"""
Acoustic signal processing — Zone B.

VAD, pause detection, phonation segment extraction, and jitter/shimmer/HNR/
pitch extraction via parselmouth (Praat). All STUBBED for now — signatures
and return shapes are fixed so /agent and ml/features.py can be built against
them immediately; fill in real parselmouth calls here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pause:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def run_vad(audio_path: str) -> list[tuple[float, float]]:
    """Voice activity detection. Returns list of (start_s, end_s) speech segments.
    TODO(ml): implement (webrtcvad / parselmouth intensity-based, etc.)."""
    raise NotImplementedError("run_vad is a stub — implement VAD.")


def detect_pauses(audio_path: str, word_timestamps: list[dict]) -> list[Pause]:
    """Pause detection from STT word-level timestamps (gaps between words)
    and/or VAD silence segments.
    `word_timestamps`: list of {"word": str, "start": float, "end": float}
    as produced by Deepgram.
    TODO(ml): implement."""
    raise NotImplementedError("detect_pauses is a stub — implement.")


def extract_phonation_segment(audio_path: str) -> str:
    """Isolate the sustained-phonation ("ahh") segment from a call recording.
    Returns path to the extracted segment (or could return array — TBD).
    TODO(ml): implement."""
    raise NotImplementedError("extract_phonation_segment is a stub — implement.")


def extract_acoustic_features(phonation_audio_path: str) -> dict:
    """Jitter/shimmer/HNR/pitch extraction via parselmouth (Praat wrapper),
    on a sustained-phonation segment. Returns a dict with keys matching
    db.contracts.FeatureVector's acoustic fields:
      jitter_local, jitter_rap, shimmer_local, shimmer_apq5, hnr
    TODO(ml): implement using parselmouth.Sound + praat call scripts, e.g.
      import parselmouth
      snd = parselmouth.Sound(phonation_audio_path)
      pitch = snd.to_pitch()
      point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
      jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
      ... etc.
    """
    raise NotImplementedError("extract_acoustic_features is a stub — implement with parselmouth.")


def extract_prosody_features(audio_path: str, word_timestamps: list[dict]) -> dict:
    """Speech rate + pause frequency/duration, from the reading/counting
    task segments. Returns dict with keys: speech_rate, pause_freq,
    pause_avg_duration (matching FeatureVector).
    TODO(ml): implement."""
    raise NotImplementedError("extract_prosody_features is a stub — implement.")
