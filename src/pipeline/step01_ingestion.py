"""Step 1: Audio Ingestion & Pre-processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

TARGET_SR = 16_000
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = TARGET_SR * VAD_FRAME_MS // 1000


@dataclass
class IngestionResult:
    samples: np.ndarray
    sample_rate: int
    duration_s: float
    speech_ratio: float
    source_path: Path


def _measure_speech_ratio(audio: np.ndarray, aggressiveness: int) -> float:
    try:
        import webrtcvad
    except ImportError:
        raise ImportError(
            "webrtcvad is required. Install via: uv add webrtcvad-wheels"
        )

    vad = webrtcvad.Vad(aggressiveness)

    remainder = len(audio) % VAD_FRAME_SAMPLES
    if remainder:
        audio = np.pad(audio, (0, VAD_FRAME_SAMPLES - remainder))

    frames = audio.reshape(-1, VAD_FRAME_SAMPLES)
    pcm_frames = (frames * 32767).clip(-32768, 32767).astype(np.int16)

    total = len(frames)
    voiced_count = 0

    for pcm_frame in pcm_frames:
        frame_bytes = pcm_frame.tobytes()
        if vad.is_speech(frame_bytes, TARGET_SR):
            voiced_count += 1

    speech_ratio = voiced_count / total if total > 0 else 0.0
    return speech_ratio


def ingest(
    path: Path | str,
    *,
    vad_aggressiveness: int = 2,
    noise_reduce: bool = True,
) -> IngestionResult:
    path = Path(path)

    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)

    if noise_reduce:
        audio = nr.reduce_noise(y=audio, sr=TARGET_SR)

    speech_ratio = _measure_speech_ratio(audio, vad_aggressiveness)
    duration_s = len(audio) / TARGET_SR

    return IngestionResult(
        samples=audio.astype(np.float32, copy=False),
        sample_rate=TARGET_SR,
        duration_s=duration_s,
        speech_ratio=speech_ratio,
        source_path=path,
    )


def save(result: IngestionResult, out_path: Path) -> None:
    sf.write(out_path, result.samples, result.sample_rate)
    meta = {
        "source_path": str(result.source_path),
        "sample_rate": result.sample_rate,
        "duration_s": result.duration_s,
        "speech_ratio": result.speech_ratio,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load(wav_path: Path) -> IngestionResult:
    samples, sample_rate = sf.read(wav_path, dtype="float32")
    meta = json.loads(wav_path.with_suffix(".json").read_text())
    return IngestionResult(
        samples=samples,
        sample_rate=sample_rate,
        duration_s=meta["duration_s"],
        speech_ratio=meta["speech_ratio"],
        source_path=Path(meta["source_path"]),
    )
