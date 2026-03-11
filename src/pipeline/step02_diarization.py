"""Step 2: Speaker Diarization."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
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
        token=token,
    )
    pipeline.to(torch.device("cpu"))

    waveform = torch.from_numpy(result.samples).unsqueeze(0)  # (1, samples)

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": result.sample_rate},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # pyannote 3.3+ may wrap result in DiarizeOutput; find the Annotation by capability
    if hasattr(diarization, "itertracks"):
        annotation = diarization
    else:
        annotation = next(
            (getattr(diarization, f) for f in dir(diarization)
             if not f.startswith("_") and hasattr(getattr(diarization, f, None), "itertracks")),
            None,
        )
        if annotation is None:
            raise RuntimeError(
                f"Cannot extract Annotation from {type(diarization).__name__}. "
                f"Available attrs: {[f for f in dir(diarization) if not f.startswith('_')]}"
            )

    segments: list[Segment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
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
