"""Step 1: Audio Ingestion & Pre-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

TARGET_SR = 16_000
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = TARGET_SR * VAD_FRAME_MS // 1000  # 320


@dataclass
class IngestionResult:
    samples: np.ndarray    # float32 audio at 16kHz
    sample_rate: int       # always 16000
    duration_s: float      # processed audio duration (after VAD)
    speech_ratio: float    # voiced_frames / total_frames
    source_path: Path


def _apply_vad(audio: np.ndarray, aggressiveness: int) -> tuple[np.ndarray, float]:
    try:
        import webrtcvad
    except ImportError:
        raise ImportError(
            "webrtcvad is required. Install via: uv add webrtcvad-wheels"
        )

    vad = webrtcvad.Vad(aggressiveness)

    # Pad to multiple of frame size
    remainder = len(audio) % VAD_FRAME_SAMPLES
    if remainder:
        audio = np.pad(audio, (0, VAD_FRAME_SAMPLES - remainder))

    frames = audio.reshape(-1, VAD_FRAME_SAMPLES)
    pcm_frames = (frames * 32767).clip(-32768, 32767).astype(np.int16)

    voiced_frames = []
    total = len(frames)
    voiced_count = 0

    for pcm_frame in pcm_frames:
        frame_bytes = pcm_frame.tobytes()
        if vad.is_speech(frame_bytes, TARGET_SR):
            voiced_frames.append(pcm_frame)
            voiced_count += 1

    speech_ratio = voiced_count / total if total > 0 else 0.0

    if voiced_frames:
        voiced_pcm = np.concatenate(voiced_frames)
        voiced_float = voiced_pcm.astype(np.float32) / 32767.0
    else:
        voiced_float = np.array([], dtype=np.float32)

    return voiced_float, speech_ratio


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

    voiced, speech_ratio = _apply_vad(audio, vad_aggressiveness)

    duration_s = len(voiced) / TARGET_SR

    return IngestionResult(
        samples=voiced,
        sample_rate=TARGET_SR,
        duration_s=duration_s,
        speech_ratio=speech_ratio,
        source_path=path,
    )


def save(result: IngestionResult, out_path: Path) -> None:
    sf.write(out_path, result.samples, result.sample_rate)
