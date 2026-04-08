from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.extraction_eval import CATEGORIES, ConsultationEvaluation, annotation_status_from_path
from evaluation.extraction_eval import consultation_from_annotation_path, evaluate_consultation, extract_predicted_items, load_gold_items


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_OUTPUTS = REPO_ROOT / "batch_outputs" / "primock57_pipeline"
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "annotations" / "extraction_gold"


def aggregate_results(results: list[ConsultationEvaluation]) -> dict[str, dict[str, float | int]]:
    """Aggregate category metrics across consultations."""
    aggregate: dict[str, dict[str, float | int]] = {}
    for category in (*CATEGORIES, "Overall"):
        aggregate[category] = {
            "gold_count": 0,
            "predicted_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "wrong_category": 0,
            "wrong_attributes": 0,
            "hallucinated": 0,
            "missed": 0,
        }

    for result in results:
        for category in CATEGORIES:
            summary = result.categories[category]
            for key in aggregate[category]:
                aggregate[category][key] += getattr(summary, key)
        overall = result.overall
        for key in aggregate["Overall"]:
            aggregate["Overall"][key] += getattr(overall, key)

    for category, values in aggregate.items():
        tp = int(values["tp"])
        fp = int(values["fp"])
        fn = int(values["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        values["precision"] = precision
        values["recall"] = recall
        values["f1"] = f1
    return aggregate


def collect_annotation_paths(
    annotations_dir: Path,
    consultations: list[str] | None,
    *,
    include_draft: bool,
) -> list[Path]:
    """Return annotation files to score."""
    paths = sorted(annotations_dir.glob("*.json"))
    if not include_draft:
        paths = [path for path in paths if annotation_status_from_path(path) in {"completed", "reviewed"}]
    if consultations is None:
        return paths
    allowed = set(consultations)
    return [path for path in paths if consultation_from_annotation_path(path) in allowed]


def print_consultation_summary(result: ConsultationEvaluation) -> None:
    """Print one consultation summary."""
    overall = result.overall
    print(
        f"{result.consultation}: "
        f"P={overall.precision:.4f} R={overall.recall:.4f} F1={overall.f1:.4f} "
        f"(gold={overall.gold_count}, predicted={overall.predicted_count}, "
        f"wrong_category={overall.wrong_category}, wrong_attributes={overall.wrong_attributes})"
    )


def print_aggregate_summary(aggregate: dict[str, dict[str, float | int]]) -> None:
    """Print the aggregate table."""
    print("\nAggregate metrics")
    for category in (*CATEGORIES, "Overall"):
        values = aggregate[category]
        print(
            f"{category}: "
            f"P={float(values['precision']):.4f} "
            f"R={float(values['recall']):.4f} "
            f"F1={float(values['f1']):.4f} "
            f"(tp={int(values['tp'])}, fp={int(values['fp'])}, fn={int(values['fn'])}, "
            f"wrong_category={int(values['wrong_category'])}, "
            f"wrong_attributes={int(values['wrong_attributes'])}, "
            f"hallucinated={int(values['hallucinated'])}, missed={int(values['missed'])})"
        )


def write_csv(path: Path, results: list[ConsultationEvaluation]) -> None:
    """Write per-consultation/category metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "consultation",
                "category",
                "gold_count",
                "predicted_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "wrong_category",
                "wrong_attributes",
                "hallucinated",
                "missed",
            ],
        )
        writer.writeheader()
        for result in results:
            for category in CATEGORIES:
                summary = result.categories[category]
                writer.writerow(
                    {
                        "consultation": result.consultation,
                        "category": category,
                        "gold_count": summary.gold_count,
                        "predicted_count": summary.predicted_count,
                        "tp": summary.tp,
                        "fp": summary.fp,
                        "fn": summary.fn,
                        "precision": f"{summary.precision:.4f}",
                        "recall": f"{summary.recall:.4f}",
                        "f1": f"{summary.f1:.4f}",
                        "wrong_category": summary.wrong_category,
                        "wrong_attributes": summary.wrong_attributes,
                        "hallucinated": summary.hallucinated,
                        "missed": summary.missed,
                    }
                )


def write_error_report(path: Path, results: list[ConsultationEvaluation]) -> None:
    """Write detailed error examples."""
    payload = []
    for result in results:
        payload.append(
            {
                "consultation": result.consultation,
                "errors": [
                    {
                        "error_type": error.error_type,
                        "category": error.category,
                        "gold_text": error.gold_text,
                        "predicted_text": error.predicted_text,
                        "score": round(error.score, 4),
                    }
                    for error in result.errors
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step 5 extraction outputs against manual gold annotations.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--batch-outputs", type=Path, default=DEFAULT_BATCH_OUTPUTS)
    parser.add_argument("--consultations", nargs="+", help="Optional consultation names to score.")
    parser.add_argument("--match-threshold", type=float, default=0.65, help="Minimum similarity needed to count as a match.")
    parser.add_argument("--output-csv", type=Path, help="Optional path for per-consultation/category metrics.")
    parser.add_argument("--error-report", type=Path, help="Optional path for detailed error examples in JSON.")
    parser.add_argument("--include-draft", action="store_true", help="Include draft annotation packets in scoring.")
    args = parser.parse_args()

    annotation_paths = collect_annotation_paths(
        args.annotations_dir,
        args.consultations,
        include_draft=args.include_draft,
    )
    if not annotation_paths:
        qualifier = "completed/reviewed" if not args.include_draft else "matching"
        raise FileNotFoundError(f"No {qualifier} annotation JSON files found in {args.annotations_dir}")

    results: list[ConsultationEvaluation] = []
    for annotation_path in annotation_paths:
        consultation, gold_items = load_gold_items(annotation_path)
        if not gold_items:
            print(f"Skipping {consultation}: no gold resources annotated yet.")
            continue
        step05_path = args.batch_outputs / consultation / f"step05_{consultation}.json"
        if not step05_path.exists():
            raise FileNotFoundError(f"Missing Step 5 output for {consultation}: {step05_path}")
        predicted_items = extract_predicted_items(step05_path)
        result = evaluate_consultation(
            consultation,
            gold_items,
            predicted_items,
            threshold=args.match_threshold,
        )
        results.append(result)
        print_consultation_summary(result)

    if not results:
        raise ValueError("No non-empty gold annotations were available to score.")

    aggregate = aggregate_results(results)
    print_aggregate_summary(aggregate)

    if args.output_csv:
        write_csv(args.output_csv, results)
    if args.error_report:
        write_error_report(args.error_report, results)


if __name__ == "__main__":
    main()
