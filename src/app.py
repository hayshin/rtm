"""Gradio web UI for the RTM clinical pipeline."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

import pipeline.step01_ingestion as step01
import pipeline.step02_diarization as step02
import pipeline.step03_transcription as step03
import pipeline.step04_postprocessing as step04
import pipeline.step05_fhir_extraction as step05
import pipeline.step06_validation as step06

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

STEP_LABELS = [
    "1. Ingestion",
    "2. Diarization",
    "3. Transcription",
    "4. Post-processing",
    "5. FHIR Extraction",
    "6. Validation",
]

PENDING = "⏸ Pending"
RUNNING = "⏳ Running…"
DONE = "✅ Done"
CACHED = "📋 Cached"
ERROR = "❌ Error"


def _status_table(rows: list[tuple[str, str, str]]) -> str:
    header = "| Step | Status | Time |\n|------|--------|------|\n"
    return header + "\n".join(f"| {l} | {s} | {t} |" for l, s, t in rows)


def _fmt_step1(r) -> str:
    return (
        f"**Duration:** {r.duration_s:.2f}s  \n**Speech ratio:** {r.speech_ratio:.3f}"
    )


def _fmt_step2(r) -> str:
    lines = [
        f"**Speakers found:** {r.num_speakers}  \n**Total segments:** {len(r.segments)}",
        "",
        "| Start | End | Speaker | Duration |",
        "|-------|-----|---------|----------|",
    ]
    for seg in r.segments[:10]:
        lines.append(
            f"| {seg.start:.2f}s | {seg.end:.2f}s | {seg.speaker} | {seg.duration:.2f}s |"
        )
    if len(r.segments) > 10:
        lines.append(f"| *…{len(r.segments) - 10} more rows* | | | |")
    return "\n".join(lines)


def _fmt_step3(r) -> str:
    lines = [
        f"**Total segments:** {len(r.segments)}",
        "",
        "| Start | End | Speaker | Text |",
        "|-------|-----|---------|------|",
    ]
    for seg in r.segments:
        text = seg.text.replace("|", "\\|")
        lines.append(f"| {seg.start:.2f}s | {seg.end:.2f}s | {seg.speaker} | {text} |")
    return "\n".join(lines)


def _fmt_step4(r) -> str:
    lines = [
        f"**Total segments:** {len(r.segments)}",
        "",
        "| Start | End | Role | Text |",
        "|-------|-----|------|------|",
    ]
    for seg in r.segments:
        text = seg.cleaned_text.replace("|", "\\|")
        lines.append(
            f"| {seg.start:.2f}s | {seg.end:.2f}s | {seg.speaker_role} | {text} |"
        )
    return "\n".join(lines)


def _fmt_step5(r) -> tuple[str, dict]:
    lines = [f"**SOAP Summary:** {r.soap_summary}", "", "**Resource Counts:**"]
    for rtype, count in r.resource_counts.items():
        lines.append(f"- {rtype}: {count}")
    return "\n".join(lines), r.bundle


def _fmt_step6(r) -> tuple[str, dict]:
    valid_str = "✅ Valid" if r.valid else "❌ Invalid"
    errors = [i for i in r.issues if i.severity == "error"]
    warnings = [i for i in r.issues if i.severity == "warning"]
    lines = [
        f"**Valid:** {valid_str}  ",
        f"**Errors:** {len(errors)}  **Warnings:** {len(warnings)}",
    ]
    if r.issues:
        lines += [
            "",
            "| Severity | Resource | ID | Message |",
            "|----------|----------|----|---------|",
        ]
        for issue in r.issues:
            msg = issue.message.replace("|", "\\|")
            lines.append(
                f"| {issue.severity} | {issue.resource_type} | {issue.resource_id} | {msg} |"
            )
    return "\n".join(lines), r.bundle_with_provenance


def _run_step1(audio_path: str, out_dir: Path):
    path = out_dir / "step01.wav"
    cached = path.exists() and path.with_suffix(".json").exists()
    t0 = time.perf_counter()
    if cached:
        return step01.load(path), True, 0.0
    result = step01.ingest(audio_path)
    step01.save(result, path)
    return result, False, time.perf_counter() - t0


def _run_step2(ingestion, out_dir: Path):
    path = out_dir / "step02.json"
    cached = path.exists()
    t0 = time.perf_counter()
    if cached:
        return step02.load(path), True, 0.0
    result = step02.diarize(ingestion)
    step02.save(result, path)
    return result, False, time.perf_counter() - t0


def _run_step3(ingestion, diarization, out_dir: Path):
    path = out_dir / "step03.json"
    cached = path.exists()
    t0 = time.perf_counter()
    if cached:
        return step03.load(path), True, 0.0
    result = step03.transcribe(ingestion, diarization)
    step03.save(result, path)
    return result, False, time.perf_counter() - t0


def _run_step4(transcription, out_dir: Path, model_id: str):
    path = out_dir / "step04.json"
    cached = path.exists()
    t0 = time.perf_counter()
    if cached:
        return step04.load(path), True, 0.0
    result = step04.postprocess(transcription, model_id=model_id)
    step04.save(result, path)
    return result, False, time.perf_counter() - t0


def _run_step5(postprocessing, out_dir: Path, model_id: str):
    path = out_dir / "step05.json"
    cached = path.exists()
    t0 = time.perf_counter()
    if cached:
        return step05.load(path), True, 0.0
    result = step05.extract(postprocessing, model_id=model_id)
    step05.save(result, path)
    return result, False, time.perf_counter() - t0


def _run_step6(extraction, out_dir: Path):
    path = out_dir / "step06.json"
    cached = path.exists()
    t0 = time.perf_counter()
    if cached:
        return step06.load(path), True, 0.0
    result = step06.validate(extraction)
    step06.save(result, path)
    return result, False, time.perf_counter() - t0


def run_pipeline(audio_path, name, step4_model, step5_model):
    """Stream status updates and outputs for each pipeline stage."""
    rows: list[tuple[str, str, str]] = [(l, PENDING, "—") for l in STEP_LABELS]
    outs: list = ["", "", "", "", "", None, "", None]

    def _yield():
        return [_status_table(rows)] + outs

    def _set_status(idx, cached, elapsed):
        rows[idx] = (
            STEP_LABELS[idx],
            CACHED if cached else DONE,
            "—" if cached else f"{elapsed:.1f}s",
        )

    if not audio_path:
        yield _yield()
        return

    name = (
        name.strip().replace(" ", "_")
        if name.strip()
        else Path(audio_path).stem.replace(" ", "_")
    )
    out_dir = OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows[0] = (STEP_LABELS[0], RUNNING, "—")
    yield _yield()
    try:
        ingestion, cached, elapsed = _run_step1(audio_path, out_dir)
        _set_status(0, cached, elapsed)
        outs[0] = _fmt_step1(ingestion)
        yield _yield()
    except Exception:
        rows[0] = (STEP_LABELS[0], ERROR, "—")
        outs[0] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    rows[1] = (STEP_LABELS[1], RUNNING, "—")
    yield _yield()
    try:
        diarization, cached, elapsed = _run_step2(ingestion, out_dir)
        _set_status(1, cached, elapsed)
        outs[1] = _fmt_step2(diarization)
        yield _yield()
    except Exception:
        rows[1] = (STEP_LABELS[1], ERROR, "—")
        outs[1] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    rows[2] = (STEP_LABELS[2], RUNNING, "—")
    yield _yield()
    try:
        transcription, cached, elapsed = _run_step3(ingestion, diarization, out_dir)
        _set_status(2, cached, elapsed)
        outs[2] = _fmt_step3(transcription)
        yield _yield()
    except Exception:
        rows[2] = (STEP_LABELS[2], ERROR, "—")
        outs[2] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    rows[3] = (STEP_LABELS[3], RUNNING, "—")
    yield _yield()
    try:
        postprocessing, cached, elapsed = _run_step4(
            transcription, out_dir, step4_model
        )
        _set_status(3, cached, elapsed)
        outs[3] = _fmt_step4(postprocessing)
        yield _yield()
    except Exception:
        rows[3] = (STEP_LABELS[3], ERROR, "—")
        outs[3] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    rows[4] = (STEP_LABELS[4], RUNNING, "—")
    yield _yield()
    try:
        extraction, cached, elapsed = _run_step5(postprocessing, out_dir, step5_model)
        _set_status(4, cached, elapsed)
        outs[4], outs[5] = _fmt_step5(extraction)
        yield _yield()
    except Exception:
        rows[4] = (STEP_LABELS[4], ERROR, "—")
        outs[4] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    rows[5] = (STEP_LABELS[5], RUNNING, "—")
    yield _yield()
    try:
        validation, cached, elapsed = _run_step6(extraction, out_dir)
        _set_status(5, cached, elapsed)
        outs[6], outs[7] = _fmt_step6(validation)
        yield _yield()
    except Exception:
        rows[5] = (STEP_LABELS[5], ERROR, "—")
        outs[6] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return


with gr.Blocks(title="RTM Pipeline") as app:
    gr.Markdown("# RTM Clinical Pipeline")

    with gr.Row():
        audio_in = gr.Audio(label="Audio file", type="filepath")
        name_in = gr.Textbox(label="Consultation name (cache key)", value="")

    with gr.Accordion("Settings", open=False):
        step4_model = gr.Textbox(label="Step 4 model", value="gpt-5-mini")
        step5_model = gr.Textbox(label="Step 5 model", value="gpt-5-mini")

    audio_in.change(
        fn=lambda p: Path(p).stem.replace(" ", "_") if p else "",
        inputs=audio_in,
        outputs=name_in,
    )

    run_btn = gr.Button("Run Pipeline", variant="primary")

    status_md = gr.Markdown()

    with gr.Accordion("Step 1 — Ingestion", open=False):
        out1 = gr.Markdown()
    with gr.Accordion("Step 2 — Diarization", open=False):
        out2 = gr.Markdown()
    with gr.Accordion("Step 3 — Transcription", open=False):
        out3 = gr.Markdown()
    with gr.Accordion("Step 4 — Post-processing", open=False):
        out4 = gr.Markdown()
    with gr.Accordion("Step 5 — FHIR Extraction", open=False):
        out5_text = gr.Markdown()
        out5_json = gr.JSON(label="FHIR Bundle")
    with gr.Accordion("Step 6 — Validation", open=False):
        out6_text = gr.Markdown()
        out6_json = gr.JSON(label="Bundle with Provenance")

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_in, name_in, step4_model, step5_model],
        outputs=[
            status_md,
            out1,
            out2,
            out3,
            out4,
            out5_text,
            out5_json,
            out6_text,
            out6_json,
        ],
    )

if __name__ == "__main__":
    app.launch()
