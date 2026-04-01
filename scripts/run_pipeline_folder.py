"""Run the full RTM pipeline for every mixed audio file in a folder.

Default input is the PriMock57 mixed dataset in ``outputs/mixed``.
Results are written to a separate root, ``batch_outputs/primock57_pipeline``,
with one subfolder per consultation so they do not collide with the
application output folders.

Usage:
    uv run python scripts/run_pipeline_folder.py
    uv run python scripts/run_pipeline_folder.py --input-dir outputs/mixed
    uv run python scripts/run_pipeline_folder.py --consultation day1_consultation01
    uv run python scripts/run_pipeline_folder.py --step4-model gpt-5 --step5-model gpt-5
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import (  # noqa: E402
    run_diarization,
    run_fhir_extraction,
    run_ingestion,
    run_postprocessing,
    run_transcription,
    run_validation,
)

DEFAULT_INPUT_DIR = ROOT / "outputs" / "mixed"
DEFAULT_OUTPUT_DIR = ROOT / "batch_outputs" / "primock57_pipeline"


def _consultation_name(audio_path: Path) -> str:
    stem = audio_path.stem
    return stem.removesuffix("_mixed")


def _find_audio_files(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.glob("*.wav") if p.is_file())
    return files


def run_pipeline_for_file(
    audio_path: Path,
    output_dir: Path,
    *,
    step4_model: str,
    step5_model: str,
) -> None:
    name = _consultation_name(audio_path)
    consultation_output_dir = output_dir / name
    consultation_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name} ===")
    print(f"Input:  {audio_path}")
    print(f"Output: {consultation_output_dir}")

    ingestion = run_ingestion(consultation_output_dir, name, audio_path)
    diarization = run_diarization(consultation_output_dir, name, ingestion)
    transcription = run_transcription(
        consultation_output_dir, name, ingestion, diarization
    )

    step04_path = consultation_output_dir / f"step04_{name}.json"
    if step04_path.exists():
        postprocessing = run_postprocessing(
            consultation_output_dir, name, transcription
        )
    else:
        print(f"Using post-processing model: {step4_model}")
        postprocessing = run_postprocessing_with_model(
            consultation_output_dir, name, transcription, step4_model
        )

    step05_path = consultation_output_dir / f"step05_{name}.json"
    if step05_path.exists():
        extraction = run_fhir_extraction(
            consultation_output_dir, name, postprocessing
        )
    else:
        print(f"Using extraction model: {step5_model}")
        extraction = run_fhir_extraction_with_model(
            consultation_output_dir, name, postprocessing, step5_model
        )

    run_validation(consultation_output_dir, name, extraction)


def run_postprocessing_with_model(output_dir: Path, name: str, transcription, model_id: str):
    import pipeline.step04_postprocessing as step04

    out = output_dir / f"step04_{name}.json"
    result = step04.postprocess(transcription, model_id=model_id)
    step04.save(result, out)
    print("\n[Post-processing]")
    print(f"Total segments: {len(result.segments)}")
    for seg in result.segments:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s] {seg.speaker_role}: {seg.cleaned_text}")
    return result


def run_fhir_extraction_with_model(output_dir: Path, name: str, postprocessing, model_id: str):
    import pipeline.step05_fhir_extraction as step05

    out = output_dir / f"step05_{name}.json"
    result = step05.extract(postprocessing, model_id=model_id)
    step05.save(result, out)
    print("\n[FHIR Extraction]")
    print(f"SOAP summary: {result.soap_summary}")
    for rtype, count in result.resource_counts.items():
        print(f"  {rtype}: {count}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RTM pipeline for every mixed audio file in a folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing mixed WAV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Separate output folder for this batch run (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--consultation",
        action="append",
        default=[],
        help="Consultation name to run. Can be repeated. Accepts values with or without the _mixed suffix.",
    )
    parser.add_argument(
        "--step4-model",
        default="gpt-5-mini",
        help="Model for transcript post-processing (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--step5-model",
        default="gpt-5-mini",
        help="Model for FHIR extraction (default: gpt-5-mini)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = _find_audio_files(args.input_dir)
    if not files:
        print(f"No .wav files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    selected = {
        name.removesuffix("_mixed")
        for name in args.consultation
    }
    if selected:
        files = [path for path in files if _consultation_name(path) in selected]
        if not files:
            print(
                f"No matching consultations found in {args.input_dir}: {sorted(selected)}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Found {len(files)} audio file(s)")
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")

    failures: list[tuple[str, str]] = []
    for audio_path in files:
        try:
            run_pipeline_for_file(
                audio_path,
                args.output_dir,
                step4_model=args.step4_model,
                step5_model=args.step5_model,
            )
        except Exception as exc:
            name = _consultation_name(audio_path)
            failures.append((name, str(exc)))
            print(f"\nERROR in {name}: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    print("\n=== Batch Summary ===")
    print(f"Succeeded: {len(files) - len(failures)}")
    print(f"Failed:    {len(failures)}")

    if failures:
        for name, message in failures:
            print(f"  {name}: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
