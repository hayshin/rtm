"""Transcript cleanup and speaker role assignment."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel

from pipeline.step03_transcription import TranscriptionResult

SYSTEM_PROMPT = """You are a medical transcription specialist. You will receive a numbered list of speaker segments from a doctor-patient consultation.

Your tasks for each segment:
1. CLEAN the text: remove filler words (um, uh, yeah, okay), fix medical homophones (ileum/ilium), expand abbreviations (BP→blood pressure), fix mis-transcribed drug/anatomy names, restructure fragmented speech into coherent utterances. Remove nonsense/dot-only segments but keep index.
2. ASSIGN speaker_role: based on the FULL transcript context, label each speaker as PHYSICIAN (asks clinical questions, examines, diagnoses, plans) or PATIENT (reports symptoms, history, experiences).

Return every segment index. Use consistent role assignments (same speaker always gets the same role)."""


class _LLMSegment(BaseModel):
    index: int
    speaker_role: str
    cleaned_text: str


class _LLMTranscript(BaseModel):
    segments: list[_LLMSegment]


@dataclass
class PostProcessedSegment:
    start: float
    end: float
    speaker: str
    speaker_role: str
    original_text: str
    cleaned_text: str
    duration: float


@dataclass
class PostProcessingResult:
    segments: list[PostProcessedSegment]
    source_path: Path


def postprocess(
    transcription: TranscriptionResult,
    *,
    model_id: str = "gpt-5-mini",
    openai_api_key: str | None = None,
) -> PostProcessingResult:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    agent = Agent(
        model=OpenAIChat(id=model_id, api_key=api_key),
        instructions=SYSTEM_PROMPT,
        output_schema=_LLMTranscript,
    )

    prompt_lines = ["Transcript segments:"]
    for i, seg in enumerate(transcription.segments):
        prompt_lines.append(
            f"[{i}] {seg.speaker} ({seg.start:.2f}s–{seg.end:.2f}s): {seg.text}"
        )

    response = agent.run("\n".join(prompt_lines))
    processed: _LLMTranscript = response.content

    by_index = {p.index: p for p in processed.segments}

    result_segments = []
    for i, seg in enumerate(transcription.segments):
        p = by_index.get(i)
        result_segments.append(
            PostProcessedSegment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                speaker_role=p.speaker_role if p else "UNKNOWN",
                original_text=seg.text,
                cleaned_text=p.cleaned_text if p else seg.text,
                duration=seg.duration,
            )
        )

    return PostProcessingResult(
        segments=result_segments,
        source_path=transcription.source_path,
    )


def save(result: PostProcessingResult, out_path: Path) -> None:
    data = {
        "source_path": str(result.source_path),
        "segments": [asdict(s) for s in result.segments],
    }
    out_path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> PostProcessingResult:
    data = json.loads(path.read_text())
    segments = [PostProcessedSegment(**s) for s in data["segments"]]
    return PostProcessingResult(
        segments=segments,
        source_path=Path(data["source_path"]),
    )
