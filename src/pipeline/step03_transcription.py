"""Step 3: ASR Transcription."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.step01_ingestion import IngestionResult
from pipeline.step02_diarization import DiarizationResult
from pipeline.runtime import resolve_compute_type, resolve_device

SAMPLE_RATE = 16_000
MIN_SEGMENT_DURATION = 0.1

BACKEND_FASTER_WHISPER = "faster-whisper"
BACKEND_TRANSFORMERS = "transformers"

_DEFAULTS = {
    BACKEND_FASTER_WHISPER: "distil-large-v3",
    BACKEND_TRANSFORMERS: "Na0s/Medical-Whisper-Large-v3",
}

_MODEL_CACHE: dict = {}


def _release_cuda_cache() -> None:
    """Release reserved CUDA memory before loading the ASR model."""
    gc.collect()

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str
    duration: float


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]
    source_path: Path


def _load_model(backend: str, model_id: str):
    key = (backend, model_id)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = resolve_device()

    if backend == BACKEND_FASTER_WHISPER:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            model_id,
            device=device,
            compute_type=resolve_compute_type(device),
        )
    elif backend == BACKEND_TRANSFORMERS:
        import torch
        from transformers import pipeline
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=0 if device == "cuda" else -1,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'faster-whisper' or 'transformers'.")

    _MODEL_CACHE[key] = model
    return model


def _transcribe_segment_faster_whisper(model, audio_slice, language: str) -> str:
    segs, _ = model.transcribe(audio_slice, language=language, beam_size=1)
    return " ".join(s.text for s in segs).strip()


def _transcribe_segment_transformers(model, audio_slice, language: str) -> str:
    result = model(audio_slice, generate_kwargs={"language": language})
    return result["text"].strip()


def transcribe(
    ingestion: IngestionResult,
    diarization: DiarizationResult,
    *,
    backend: str = BACKEND_TRANSFORMERS,
    model_id: str | None = None,
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe diarized segments with the selected backend and model."""
    resolved_model = model_id or _DEFAULTS[backend]
    _release_cuda_cache()
    model = _load_model(backend, resolved_model)

    if backend == BACKEND_FASTER_WHISPER:
        _transcribe = _transcribe_segment_faster_whisper
    else:
        _transcribe = _transcribe_segment_transformers

    segments: list[TranscriptSegment] = []

    for seg in diarization.segments:
        if seg.duration < MIN_SEGMENT_DURATION:
            continue

        start_sample = int(seg.start * SAMPLE_RATE)
        end_sample = int(seg.end * SAMPLE_RATE)
        audio_slice = ingestion.samples[start_sample:end_sample]

        text = _transcribe(model, audio_slice, language)

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
