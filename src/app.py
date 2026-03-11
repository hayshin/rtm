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
DONE    = "✅ Done"
CACHED  = "📋 Cached"
ERROR   = "❌ Error"


# ── Status table ──────────────────────────────────────────────────────────────

def _status_table(rows: list[tuple[str, str, str]]) -> str:
    header = "| Step | Status | Time |\n|------|--------|------|\n"
    return header + "\n".join(f"| {l} | {s} | {t} |" for l, s, t in rows)


# ── Output formatters ─────────────────────────────────────────────────────────

def _fmt_step1(r) -> str:
    return f"**Duration:** {r.duration_s:.2f}s  \n**Speech ratio:** {r.speech_ratio:.3f}"


def _fmt_step2(r) -> str:
    lines = [
        f"**Speakers found:** {r.num_speakers}  \n**Total segments:** {len(r.segments)}",
        "",
        "| Start | End | Speaker | Duration |",
        "|-------|-----|---------|----------|",
    ]
    for seg in r.segments[:10]:
        lines.append(f"| {seg.start:.2f}s | {seg.end:.2f}s | {seg.speaker} | {seg.duration:.2f}s |")
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
        lines.append(f"| {seg.start:.2f}s | {seg.end:.2f}s | {seg.speaker_role} | {text} |")
    return "\n".join(lines)


def _fmt_step5(r) -> tuple[str, dict]:
    lines = [f"**SOAP Summary:** {r.soap_summary}", "", "**Resource Counts:**"]
    for rtype, count in r.resource_counts.items():
        lines.append(f"- {rtype}: {count}")
    return "\n".join(lines), r.bundle


def _fmt_step6(r) -> tuple[str, dict]:
    valid_str = "✅ Valid" if r.valid else "❌ Invalid"
    errors   = [i for i in r.issues if i.severity == "error"]
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


# ── Generator ─────────────────────────────────────────────────────────────────

def run_pipeline(audio_path, name, step4_model, step5_model):
    """Generator: yields [status_md, out1, out2, out3, out4, out5_text, out5_json, out6_text, out6_json]."""
    rows: list[tuple[str, str, str]] = [(l, PENDING, "—") for l in STEP_LABELS]
    # outs indices: 0=out1, 1=out2, 2=out3, 3=out4, 4=out5_text, 5=out5_json, 6=out6_text, 7=out6_json
    outs: list = ["", "", "", "", "", None, "", None]

    def _yield():
        return [_status_table(rows)] + outs

    if not audio_path:
        yield _yield()
        return

    name = (name.strip().replace(" ", "_") if name.strip()
            else Path(audio_path).stem.replace(" ", "_"))
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── Step 1: Ingestion ──────────────────────────────────────────────────────
    rows[0] = (STEP_LABELS[0], RUNNING, "—")
    yield _yield()

    step1_wav = OUTPUTS_DIR / f"step01_{name}.wav"
    is_cached = step1_wav.exists() and step1_wav.with_suffix(".json").exists()
    try:
        t0 = time.perf_counter()
        if is_cached:
            ingestion = step01.load(step1_wav)
            rows[0] = (STEP_LABELS[0], CACHED, "—")
        else:
            ingestion = step01.ingest(audio_path)
            step01.save(ingestion, step1_wav)
            rows[0] = (STEP_LABELS[0], DONE, f"{time.perf_counter() - t0:.1f}s")
        outs[0] = _fmt_step1(ingestion)
        yield _yield()
    except Exception:
        rows[0] = (STEP_LABELS[0], ERROR, "—")
        outs[0] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    # ── Step 2: Diarization ────────────────────────────────────────────────────
    rows[1] = (STEP_LABELS[1], RUNNING, "—")
    yield _yield()

    step2_json = OUTPUTS_DIR / f"step02_{name}.json"
    is_cached = step2_json.exists()
    try:
        t0 = time.perf_counter()
        if is_cached:
            diarization = step02.load(step2_json)
            rows[1] = (STEP_LABELS[1], CACHED, "—")
        else:
            diarization = step02.diarize(ingestion)
            step02.save(diarization, step2_json)
            rows[1] = (STEP_LABELS[1], DONE, f"{time.perf_counter() - t0:.1f}s")
        outs[1] = _fmt_step2(diarization)
        yield _yield()
    except Exception:
        rows[1] = (STEP_LABELS[1], ERROR, "—")
        outs[1] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    # ── Step 3: Transcription ──────────────────────────────────────────────────
    rows[2] = (STEP_LABELS[2], RUNNING, "—")
    yield _yield()

    step3_json = OUTPUTS_DIR / f"step03_{name}.json"
    is_cached = step3_json.exists()
    try:
        t0 = time.perf_counter()
        if is_cached:
            transcription = step03.load(step3_json)
            rows[2] = (STEP_LABELS[2], CACHED, "—")
        else:
            transcription = step03.transcribe(ingestion, diarization)
            step03.save(transcription, step3_json)
            rows[2] = (STEP_LABELS[2], DONE, f"{time.perf_counter() - t0:.1f}s")
        outs[2] = _fmt_step3(transcription)
        yield _yield()
    except Exception:
        rows[2] = (STEP_LABELS[2], ERROR, "—")
        outs[2] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    # ── Step 4: Post-processing ────────────────────────────────────────────────
    rows[3] = (STEP_LABELS[3], RUNNING, "—")
    yield _yield()

    step4_json = OUTPUTS_DIR / f"step04_{name}.json"
    is_cached = step4_json.exists()
    try:
        t0 = time.perf_counter()
        if is_cached:
            postprocessing = step04.load(step4_json)
            rows[3] = (STEP_LABELS[3], CACHED, "—")
        else:
            postprocessing = step04.postprocess(transcription, model_id=step4_model)
            step04.save(postprocessing, step4_json)
            rows[3] = (STEP_LABELS[3], DONE, f"{time.perf_counter() - t0:.1f}s")
        outs[3] = _fmt_step4(postprocessing)
        yield _yield()
    except Exception:
        rows[3] = (STEP_LABELS[3], ERROR, "—")
        outs[3] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    # ── Step 5: FHIR Extraction ────────────────────────────────────────────────
    rows[4] = (STEP_LABELS[4], RUNNING, "—")
    yield _yield()

    step5_json = OUTPUTS_DIR / f"step05_{name}.json"
    is_cached = step5_json.exists()
    try:
        t0 = time.perf_counter()
        if is_cached:
            extraction = step05.load(step5_json)
            rows[4] = (STEP_LABELS[4], CACHED, "—")
        else:
            extraction = step05.extract(postprocessing, model_id=step5_model)
            step05.save(extraction, step5_json)
            rows[4] = (STEP_LABELS[4], DONE, f"{time.perf_counter() - t0:.1f}s")
        outs[4], outs[5] = _fmt_step5(extraction)
        yield _yield()
    except Exception:
        rows[4] = (STEP_LABELS[4], ERROR, "—")
        outs[4] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return

    # ── Step 6: Validation ─────────────────────────────────────────────────────
    rows[5] = (STEP_LABELS[5], RUNNING, "—")
    yield _yield()

    step6_json = OUTPUTS_DIR / f"step06_{name}.json"
    is_cached = step6_json.exists()
    try:
        t0 = time.perf_counter()
        if is_cached:
            validation = step06.load(step6_json)
            rows[5] = (STEP_LABELS[5], CACHED, "—")
        else:
            validation = step06.validate(extraction)
            step06.save(validation, step6_json)
            rows[5] = (STEP_LABELS[5], DONE, f"{time.perf_counter() - t0:.1f}s")
        outs[6], outs[7] = _fmt_step6(validation)
        yield _yield()
    except Exception:
        rows[5] = (STEP_LABELS[5], ERROR, "—")
        outs[6] = f"**Error:**\n```\n{traceback.format_exc()}\n```"
        yield _yield()
        return


# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(title="RTM Pipeline") as app:
    gr.Markdown("# RTM Clinical Pipeline")

    with gr.Row():
        audio_in = gr.Audio(label="Audio file", type="filepath")
        name_in  = gr.Textbox(label="Consultation name (cache key)", value="ui_session")

    with gr.Accordion("Settings", open=False):
        step4_model = gr.Textbox(label="Step 4 model", value="gpt-4o-mini")
        step5_model = gr.Textbox(label="Step 5 model", value="gpt-4o-mini")

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
        outputs=[status_md, out1, out2, out3, out4, out5_text, out5_json, out6_text, out6_json],
    )

if __name__ == "__main__":
    app.launch()
