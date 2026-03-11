"""Step 3: ASR Transcription."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.step01_ingestion import IngestionResult
from pipeline.step02_diarization import DiarizationResult

SAMPLE_RATE = 16_000
MIN_SEGMENT_DURATION = 0.1  # seconds


@dataclass
class TranscriptSegment:
    start: float    # seconds (from diarization)
    end: float      # seconds
    speaker: str    # "SPEAKER_00", "SPEAKER_01"
    text: str       # transcribed text
    duration: float # end - start


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]  # sorted by start time
    source_path: Path


def transcribe(
    ingestion: IngestionResult,
    diarization: DiarizationResult,
    *,
    model_id: str = "Na0s/Medical-Whisper-Large-v3",
    language: str = "en",
) -> TranscriptionResult:
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers is required. Install via: uv add 'transformers>=4.40'"
        )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        dtype=torch.float32,
        device="cpu",
    )

    segments: list[TranscriptSegment] = []

    for seg in diarization.segments:
        if seg.duration < MIN_SEGMENT_DURATION:
            continue

        start_sample = int(seg.start * SAMPLE_RATE)
        end_sample = int(seg.end * SAMPLE_RATE)
        audio_slice = ingestion.samples[start_sample:end_sample]

        result = pipe(audio_slice, generate_kwargs={"language": language})
        text = result["text"].strip()

        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            speaker=seg.speaker,
            text=text,
            duration=seg.duration,
        ))

    segments.sort(key=lambda s: s.start)

    return TranscriptionResult(
        segments=segments,
        source_path=ingestion.source_path,
    )


def save(result: TranscriptionResult, out_path: Path) -> None:
    data = {
        "source_path": str(result.source_path),
        "segments": [asdict(s) for s in result.segments],
    }
    out_path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> TranscriptionResult:
    data = json.loads(path.read_text())
    segments = [TranscriptSegment(**s) for s in data["segments"]]
    return TranscriptionResult(
        segments=segments,
        source_path=Path(data["source_path"]),
    )
