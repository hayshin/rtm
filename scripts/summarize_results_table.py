"""Summarize aggregate metrics for the results table in research/results.tex.

By default this script reads the batch pipeline outputs from
``batch_outputs/primock57_pipeline`` and prints the numeric values used in the
aggregate results table. It can emit either plain text, JSON, or LaTeX table
rows for direct reuse in the paper.

Usage:
    python scripts/summarize_results_table.py
    python scripts/summarize_results_table.py --format json
    python scripts/summarize_results_table.py --format latex
    python scripts/summarize_results_table.py --input-dir batch_outputs/primock57_pipeline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "batch_outputs" / "primock57_pipeline"


@dataclass
class Summary:
    completed_consultations: int
    validation_passes: int
    validation_issues: int
    total_duration_s: float
    mean_duration_s: float
    mean_speech_ratio: float
    min_speech_ratio: float
    max_speech_ratio: float
    mean_diarized_segments_ge_0_1s: float
    encounter_resources: int
    condition_resources: int
    medication_statement_resources: int
    observation_resources: int
    procedure_resources: int
    provenance_resources: int


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def consultation_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def build_summary(input_dir: Path) -> Summary:
    dirs = consultation_dirs(input_dir)
    if not dirs:
        raise FileNotFoundError(f"No consultation directories found in {input_dir}")

    total_duration_s = 0.0
    total_speech_ratio = 0.0
    speech_ratios: list[float] = []
    total_segment_count = 0
    validation_passes = 0
    validation_issues = 0
    resource_totals: Counter[str] = Counter()
    provenance_resources = 0

    for consultation_dir in dirs:
        name = consultation_dir.name
        step01 = load_json(consultation_dir / f"step01_{name}.json")
        step02 = load_json(consultation_dir / f"step02_{name}.json")
        step06 = load_json(consultation_dir / f"step06_{name}.json")

        total_duration_s += float(step01["duration_s"])
        speech_ratio = float(step01["speech_ratio"])
        total_speech_ratio += speech_ratio
        speech_ratios.append(speech_ratio)

        diarized_segments = [
            segment
            for segment in step02["segments"]
            if float(segment["duration"]) >= 0.1
        ]
        total_segment_count += len(diarized_segments)

        if bool(step06["valid"]):
            validation_passes += 1
        validation_issues += len(step06["issues"])
        resource_totals.update(step06["resource_counts"])
        provenance_resources += sum(
            1
            for entry in step06["bundle_with_provenance"]["entry"]
            if entry["resource"]["resourceType"] == "Provenance"
        )

    completed = len(dirs)
    return Summary(
        completed_consultations=completed,
        validation_passes=validation_passes,
        validation_issues=validation_issues,
        total_duration_s=round(total_duration_s, 2),
        mean_duration_s=round(total_duration_s / completed, 2),
        mean_speech_ratio=round(total_speech_ratio / completed, 3),
        min_speech_ratio=round(min(speech_ratios), 3),
        max_speech_ratio=round(max(speech_ratios), 3),
        mean_diarized_segments_ge_0_1s=round(total_segment_count / completed, 1),
        encounter_resources=resource_totals["Encounter"],
        condition_resources=resource_totals["Condition"],
        medication_statement_resources=resource_totals["MedicationStatement"],
        observation_resources=resource_totals["Observation"],
        procedure_resources=resource_totals["Procedure"],
        provenance_resources=provenance_resources,
    )


def render_text(summary: Summary) -> str:
    return "\n".join(
        [
            f"Completed consultations: {summary.completed_consultations}",
            (
                "Validation pass rate: "
                f"{summary.validation_passes}/{summary.completed_consultations} "
                f"({summary.validation_passes / summary.completed_consultations:.0%})"
            ),
            f"Validation issues: {summary.validation_issues}",
            f"Total duration (s): {summary.total_duration_s:.2f}",
            f"Mean duration (s): {summary.mean_duration_s:.2f}",
            f"Mean speech ratio: {summary.mean_speech_ratio:.3f}",
            (
                "Speech-ratio range: "
                f"{summary.min_speech_ratio:.3f}--{summary.max_speech_ratio:.3f}"
            ),
            (
                "Mean diarized segments (>= 0.1 s): "
                f"{summary.mean_diarized_segments_ge_0_1s:.1f}"
            ),
            f"Encounter resources: {summary.encounter_resources}",
            f"Condition resources: {summary.condition_resources}",
            (
                "MedicationStatement resources: "
                f"{summary.medication_statement_resources}"
            ),
            f"Observation resources: {summary.observation_resources}",
            f"Procedure resources: {summary.procedure_resources}",
            f"Provenance resources: {summary.provenance_resources}",
        ]
    )


def render_latex(summary: Summary) -> str:
    rows = [
        ("Completed consultations", f"{summary.completed_consultations}"),
        (
            "Validation pass rate",
            (
                f"{summary.validation_passes}/{summary.completed_consultations} "
                f"({summary.validation_passes / summary.completed_consultations:.0%})"
            ),
        ),
        ("Validation issues", f"{summary.validation_issues}"),
        ("Total duration (s)", f"{summary.total_duration_s:.2f}"),
        ("Mean duration (s)", f"{summary.mean_duration_s:.2f}"),
        ("Mean speech ratio", f"{summary.mean_speech_ratio:.3f}"),
        (
            "Speech-ratio range",
            f"{summary.min_speech_ratio:.3f}--{summary.max_speech_ratio:.3f}",
        ),
        (
            "Mean diarized segments ($\\geq 0.1$\\,s)",
            f"{summary.mean_diarized_segments_ge_0_1s:.1f}",
        ),
        ("\\texttt{Encounter} resources", f"{summary.encounter_resources}"),
        ("\\texttt{Condition} resources", f"{summary.condition_resources}"),
        (
            "\\texttt{MedicationStatement} resources",
            f"{summary.medication_statement_resources}",
        ),
        ("\\texttt{Observation} resources", f"{summary.observation_resources}"),
        ("\\texttt{Procedure} resources", f"{summary.procedure_resources}"),
        ("\\texttt{Provenance} resources", f"{summary.provenance_resources}"),
    ]
    return "\n".join(f"{label} & {value} \\\\" for label, value in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize aggregate metrics for the results table."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Pipeline batch output directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "latex"),
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    summary = build_summary(args.input_dir)

    if args.format == "json":
        print(json.dumps(asdict(summary), indent=2))
        return
    if args.format == "latex":
        print(render_latex(summary))
        return
    print(render_text(summary))


if __name__ == "__main__":
    main()
