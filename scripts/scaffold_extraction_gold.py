from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.extraction_eval import CATEGORIES, predicted_candidates_for_annotation, transcript_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_OUTPUTS = REPO_ROOT / "batch_outputs" / "primock57_pipeline"
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "annotations" / "extraction_gold"


def collect_consultations(batch_outputs: Path, consultation_names: list[str] | None, limit: int) -> list[str]:
    """Return the consultations to scaffold."""
    if consultation_names:
        return consultation_names
    names = sorted(path.name for path in batch_outputs.iterdir() if path.is_dir())
    return names[:limit]


def empty_gold_resources() -> dict[str, list[dict[str, object]]]:
    """Return the editable gold resource scaffold."""
    return {category: [] for category in CATEGORIES}


def build_annotation_payload(consultation: str, transcript_path: Path, prediction_path: Path) -> dict[str, object]:
    """Build one annotation JSON document."""
    return {
        "schema_version": 1,
        "consultation": consultation,
        "annotation_status": "draft",
        "annotator": "",
        "notes": "",
        "transcript_packet": transcript_path.name,
        "prediction_path": str(prediction_path),
        "instructions": (
            "Fill gold_resources only. Use segment_indices from the transcript packet. "
            "predicted_candidates are context only and are ignored by the evaluator."
        ),
        "gold_resources": empty_gold_resources(),
        "predicted_candidates": predicted_candidates_for_annotation(prediction_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manual extraction annotation packets from Step 4 and Step 5 outputs.")
    parser.add_argument("--batch-outputs", type=Path, default=DEFAULT_BATCH_OUTPUTS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--consultations", nargs="+", help="Explicit consultation names to scaffold.")
    parser.add_argument("--limit", type=int, default=5, help="How many consultations to scaffold if --consultations is omitted.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation packets.")
    args = parser.parse_args()

    consultations = collect_consultations(args.batch_outputs, args.consultations, args.limit)
    args.annotations_dir.mkdir(parents=True, exist_ok=True)

    for consultation in consultations:
        consultation_dir = args.batch_outputs / consultation
        step04_path = consultation_dir / f"step04_{consultation}.json"
        step05_path = consultation_dir / f"step05_{consultation}.json"
        if not step04_path.exists() or not step05_path.exists():
            raise FileNotFoundError(f"Missing step04/step05 outputs for {consultation}")

        transcript_path = args.annotations_dir / f"{consultation}_transcript.md"
        annotation_path = args.annotations_dir / f"{consultation}.json"

        if args.overwrite or not transcript_path.exists():
            transcript_path.write_text(transcript_markdown(step04_path, consultation))

        if args.overwrite or not annotation_path.exists():
            payload = build_annotation_payload(consultation, transcript_path, step05_path)
            annotation_path.write_text(json.dumps(payload, indent=2))

        print(f"Scaffolded {consultation}: {annotation_path}")


if __name__ == "__main__":
    main()
