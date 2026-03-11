"""Step 2: Speaker Diarization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.step01_ingestion import IngestionResult


@dataclass
class Segment:
    start: float    # seconds
    end: float      # seconds
    speaker: str    # "SPEAKER_00", "SPEAKER_01", ...
    duration: float # end - start


@dataclass
class DiarizationResult:
    segments: list[Segment]   # sorted by start time
    num_speakers: int
    source_path: Path


def diarize(
    result: IngestionResult,
    *,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 2,
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

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    pipeline.to(torch.device("cpu"))

    waveform = torch.from_numpy(result.samples).unsqueeze(0)  # (1, samples)

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": result.sample_rate},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments: list[Segment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
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
