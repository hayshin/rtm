"""Step 2: Speaker Diarization."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa

from pipeline.step01_ingestion import IngestionResult
from pipeline.runtime import resolve_device


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    duration: float


@dataclass
class DiarizationResult:
    segments: list[Segment]
    num_speakers: int
    source_path: Path


def diarize(
    result: IngestionResult,
    *,
    num_speakers: int = 2,
    hf_token: str | None = None,
) -> DiarizationResult:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError(
            "pyannote.audio is required. Install via: uv add 'pyannote.audio>=3.3'"
        )

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass hf_token=."
        )

    device = resolve_device()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    pipeline.to(torch.device(device))

    raw_audio, sample_rate = librosa.load(result.source_path, sr=16_000, mono=True)
    waveform = torch.from_numpy(raw_audio).unsqueeze(0)

    output = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=num_speakers,
    )

    segments: list[Segment] = []
    for turn, speaker in output.speaker_diarization:
        segments.append(Segment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
            duration=turn.end - turn.start,
        ))

    segments.sort(key=lambda s: s.start)
    num_speakers_found = len({s.speaker for s in segments})

    return DiarizationResult(
        segments=segments,
        num_speakers=num_speakers_found,
        source_path=result.source_path,
    )


def save(result: DiarizationResult, out_path: Path) -> None:
    data = {
        "source_path": str(result.source_path),
        "num_speakers": result.num_speakers,
        "segments": [asdict(s) for s in result.segments],
    }
    out_path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> DiarizationResult:
    data = json.loads(path.read_text())
    segments = [Segment(**s) for s in data["segments"]]
    return DiarizationResult(
        segments=segments,
        num_speakers=data["num_speakers"],
        source_path=Path(data["source_path"]),
    )
