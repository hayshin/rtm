from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_OUTPUTS = REPO_ROOT / "batch_outputs" / "primock57_pipeline"
DEFAULT_TRANSCRIPTS = REPO_ROOT / "references" / "primock57" / "transcripts"
TRANSCRIPT_TAGS = ("<UNSURE>", "</UNSURE>", "<UNIN/>", "<INAUDIBLE_SPEECH/>")


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


def strip_transcript_tags(text: str) -> str:
    for tag in TRANSCRIPT_TAGS:
        text = text.replace(tag, "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = strip_transcript_tags(text).lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9 ']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein_length(seq1: list[str], seq2: list[str]) -> int:
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    prev = list(range(len(seq2) + 1))
    for i, left in enumerate(seq1, start=1):
        curr = [i]
        for j, right in enumerate(seq2, start=1):
            cost = 0 if left == right else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens:
        return (0.0 if not hyp_tokens else 1.0, 0, len(hyp_tokens))
    distance = levenshtein_length(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens), distance, len(ref_tokens)


def char_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return (0.0 if not hyp_chars else 1.0, 0, len(hyp_chars))
    distance = levenshtein_length(ref_chars, hyp_chars)
    return distance / len(ref_chars), distance, len(ref_chars)


def parse_textgrid_intervals(path: Path, speaker: str) -> list[Utterance]:
    lines = path.read_text().splitlines()
    utterances: list[Utterance] = []
    xmin: float | None = None
    xmax: float | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("xmin = "):
            xmin = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax = "):
            xmax = float(line.split("=", 1)[1].strip())
        elif line.startswith("text = "):
            text = line.split("=", 1)[1].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            cleaned = strip_transcript_tags(text)
            if xmin is not None and xmax is not None and cleaned:
                utterances.append(Utterance(speaker=speaker, start=xmin, end=xmax, text=cleaned))
    return utterances


def load_reference_utterances(consultation: str, transcripts_dir: Path) -> list[Utterance]:
    doctor_path = transcripts_dir / f"{consultation}_doctor.TextGrid"
    patient_path = transcripts_dir / f"{consultation}_patient.TextGrid"
    if not doctor_path.exists() or not patient_path.exists():
        raise FileNotFoundError(f"Missing PriMock57 transcripts for {consultation}")

    utterances = parse_textgrid_intervals(doctor_path, "PHYSICIAN")
    utterances.extend(parse_textgrid_intervals(patient_path, "PATIENT"))
    utterances.sort(key=lambda item: (item.start, item.end, item.speaker))
    return utterances


def merged_reference_text(utterances: Iterable[Utterance]) -> str:
    return " ".join(item.text for item in utterances if item.text.strip())


def load_prediction_text(consultation_dir: Path, source: str) -> str:
    step_name = "step03" if source == "step03" else "step04"
    path = consultation_dir / f"{step_name}_{consultation_dir.name}.json"
    payload = json.loads(path.read_text())

    if source == "step03":
        parts = [segment["text"] for segment in payload["segments"] if segment["text"].strip()]
    else:
        parts = [segment["cleaned_text"] for segment in payload["segments"] if segment["cleaned_text"].strip()]
    return " ".join(parts)


def best_overlap_role(start: float, end: float, reference: Iterable[Utterance]) -> str | None:
    overlaps: Counter[str] = Counter()
    for item in reference:
        overlap = max(0.0, min(end, item.end) - max(start, item.start))
        if overlap > 0:
            overlaps[item.speaker] += overlap
    if not overlaps:
        return None
    return overlaps.most_common(1)[0][0]


def speaker_role_metrics(consultation_dir: Path, reference: list[Utterance]) -> tuple[float | None, Counter[tuple[str, str]], int]:
    path = consultation_dir / f"step04_{consultation_dir.name}.json"
    if not path.exists():
        return None, Counter(), 0

    payload = json.loads(path.read_text())
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    total = 0

    for segment in payload["segments"]:
        predicted = segment.get("speaker_role")
        cleaned = segment.get("cleaned_text", "").strip()
        if predicted not in {"PHYSICIAN", "PATIENT"} or not cleaned:
            continue

        reference_role = best_overlap_role(segment["start"], segment["end"], reference)
        if reference_role is None:
            continue

        confusion[(reference_role, predicted)] += 1
        correct += int(reference_role == predicted)
        total += 1

    if total == 0:
        return None, confusion, total
    return correct / total, confusion, total


def collect_consultation_dirs(batch_outputs: Path) -> list[Path]:
    return sorted(path for path in batch_outputs.iterdir() if path.is_dir())


def format_rate(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PriMock57 transcript outputs against reference TextGrids.")
    parser.add_argument("--batch-outputs", type=Path, default=DEFAULT_BATCH_OUTPUTS)
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["step03", "step04"],
        default=["step03", "step04"],
        help="Prediction sources to score. step04 uses cleaned_text.",
    )
    parser.add_argument(
        "--include-cer",
        action="store_true",
        help="Also compute CER. This is slower than WER on long consultations.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional path for per-consultation metrics.")
    args = parser.parse_args()

    consultation_dirs = collect_consultation_dirs(args.batch_outputs)
    rows: list[dict[str, str]] = []
    aggregate_word: dict[str, tuple[int, int]] = {source: (0, 0) for source in args.sources}
    aggregate_char: dict[str, tuple[int, int]] = {source: (0, 0) for source in args.sources}
    aggregate_confusion: Counter[tuple[str, str]] = Counter()
    aggregate_role_correct = 0
    aggregate_role_total = 0

    for consultation_dir in consultation_dirs:
        consultation = consultation_dir.name
        reference_utterances = load_reference_utterances(consultation, args.transcripts_dir)
        reference_text = merged_reference_text(reference_utterances)

        role_accuracy, confusion, role_total = speaker_role_metrics(consultation_dir, reference_utterances)
        if role_accuracy is not None:
            aggregate_confusion.update(confusion)
            aggregate_role_total += role_total
            aggregate_role_correct += sum(
                count for (reference_role, predicted_role), count in confusion.items() if reference_role == predicted_role
            )

        for source in args.sources:
            prediction_text = load_prediction_text(consultation_dir, source)
            wer, word_distance, word_ref_len = word_error_rate(reference_text, prediction_text)
            word_num, word_den = aggregate_word[source]
            aggregate_word[source] = (word_num + word_distance, word_den + word_ref_len)

            cer_str = ""
            if args.include_cer:
                cer, char_distance, char_ref_len = char_error_rate(reference_text, prediction_text)
                char_num, char_den = aggregate_char[source]
                aggregate_char[source] = (char_num + char_distance, char_den + char_ref_len)
                cer_str = f"{cer:.4f}"

            rows.append(
                {
                    "consultation": consultation,
                    "source": source,
                    "wer": f"{wer:.4f}",
                    "cer": cer_str,
                    "speaker_role_accuracy": format_rate(role_accuracy) if source == "step04" else "",
                    "role_segments_scored": str(role_total) if source == "step04" and role_accuracy is not None else "",
                }
            )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "consultation",
                    "source",
                    "wer",
                    "cer",
                    "speaker_role_accuracy",
                    "role_segments_scored",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("Per-consultation metrics")
    for row in rows:
        role_suffix = ""
        if row["source"] == "step04" and row["speaker_role_accuracy"]:
            role_suffix = (
                f", role_accuracy={row['speaker_role_accuracy']}"
                f", role_segments={row['role_segments_scored']}"
            )
        print(
            f"{row['consultation']} {row['source']}: "
            f"WER={row['wer']}, CER={row['cer']}{role_suffix}"
        )

    print("\nSummary")
    for source in args.sources:
        word_distance, word_ref_len = aggregate_word[source]
        summary_wer = word_distance / word_ref_len if word_ref_len else 0.0
        if args.include_cer:
            char_distance, char_ref_len = aggregate_char[source]
            summary_cer = char_distance / char_ref_len if char_ref_len else 0.0
            print(f"{source}: WER={summary_wer:.4f}, CER={summary_cer:.4f}")
        else:
            print(f"{source}: WER={summary_wer:.4f}")

    if aggregate_role_total:
        overall_role_accuracy = aggregate_role_correct / aggregate_role_total
        print(f"\nSpeaker-role summary: accuracy={overall_role_accuracy:.4f}, segments={aggregate_role_total}")
        print("Confusion matrix (reference -> predicted)")
        for reference_role in ("PHYSICIAN", "PATIENT"):
            for predicted_role in ("PHYSICIAN", "PATIENT"):
                count = aggregate_confusion[(reference_role, predicted_role)]
                print(f"{reference_role} -> {predicted_role}: {count}")
    else:
        print("\nSpeaker-role summary: no scoreable step04 segments found.")


if __name__ == "__main__":
    main()
