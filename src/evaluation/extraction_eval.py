from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


CATEGORIES = ("Condition", "MedicationStatement", "Observation", "Procedure")


def normalize_text(text: str) -> str:
    """Normalize free text for loose semantic matching."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    """Return normalized tokens for overlap scoring."""
    normalized = normalize_text(text)
    return set(normalized.split()) if normalized else set()


def token_f1(left: str, right: str) -> float:
    """Compute token-set F1."""
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    return 2 * precision * recall / (precision + recall)


def jaccard(left: set[int], right: set[int]) -> float:
    """Compute Jaccard overlap for segment indices."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def metric_triplet(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return precision, recall, and F1."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return precision, recall, f1


def parse_segment_indices(value: str | None) -> list[int]:
    """Parse the custom FHIR extension valueString into sorted indices."""
    if not value:
        return []
    indices: list[int] = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        indices.append(int(stripped))
    return sorted(set(indices))


@dataclass(frozen=True)
class ExtractionItem:
    category: str
    text: str
    segment_indices: tuple[int, ...]
    attributes: dict[str, str]
    source: str


@dataclass(frozen=True)
class MatchResult:
    gold: ExtractionItem
    predicted: ExtractionItem
    score: float


@dataclass(frozen=True)
class ErrorRecord:
    consultation: str
    error_type: str
    category: str
    gold_text: str
    predicted_text: str
    score: float


@dataclass(frozen=True)
class CategorySummary:
    gold_count: int
    predicted_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    wrong_category: int
    wrong_attributes: int
    hallucinated: int
    missed: int


@dataclass(frozen=True)
class ConsultationEvaluation:
    consultation: str
    categories: dict[str, CategorySummary]
    overall: CategorySummary
    errors: list[ErrorRecord]


def item_match_score(gold: ExtractionItem, predicted: ExtractionItem) -> float:
    """Compute a loose semantic similarity score between gold and prediction."""
    text_score = token_f1(gold.text, predicted.text)
    left_segments = set(gold.segment_indices)
    right_segments = set(predicted.segment_indices)
    if left_segments or right_segments:
        return (0.7 * text_score) + (0.3 * jaccard(left_segments, right_segments))
    return text_score


def normalize_attribute(value: str) -> str:
    """Normalize attribute strings before comparison."""
    return normalize_text(value)


def attribute_mismatch_count(gold: ExtractionItem, predicted: ExtractionItem) -> int:
    """Count mismatched annotated attributes on a matched pair."""
    mismatches = 0
    for key, gold_value in gold.attributes.items():
        if not gold_value.strip():
            continue
        predicted_value = predicted.attributes.get(key, "")
        if normalize_attribute(gold_value) != normalize_attribute(predicted_value):
            mismatches += 1
    return mismatches


def greedy_match(
    gold_items: list[ExtractionItem],
    predicted_items: list[ExtractionItem],
    *,
    threshold: float,
) -> tuple[list[MatchResult], list[ExtractionItem], list[ExtractionItem]]:
    """Greedily match pairs by descending similarity."""
    candidates: list[tuple[float, int, int]] = []
    for gold_index, predicted_index in product(range(len(gold_items)), range(len(predicted_items))):
        score = item_match_score(gold_items[gold_index], predicted_items[predicted_index])
        if score >= threshold:
            candidates.append((score, gold_index, predicted_index))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_gold: set[int] = set()
    matched_predicted: set[int] = set()
    matches: list[MatchResult] = []

    for score, gold_index, predicted_index in candidates:
        if gold_index in matched_gold or predicted_index in matched_predicted:
            continue
        matched_gold.add(gold_index)
        matched_predicted.add(predicted_index)
        matches.append(
            MatchResult(
                gold=gold_items[gold_index],
                predicted=predicted_items[predicted_index],
                score=score,
            )
        )

    unmatched_gold = [item for index, item in enumerate(gold_items) if index not in matched_gold]
    unmatched_predicted = [item for index, item in enumerate(predicted_items) if index not in matched_predicted]
    return matches, unmatched_gold, unmatched_predicted


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text())


def extract_predicted_items(step05_path: Path) -> list[ExtractionItem]:
    """Convert a saved Step 5 bundle into evaluation items."""
    payload = read_json(step05_path)
    items: list[ExtractionItem] = []
    for entry in payload["bundle"]["entry"]:
        resource = entry["resource"]
        category = resource["resourceType"]
        if category not in CATEGORIES:
            continue

        attributes: dict[str, str]
        text: str
        if category == "Condition":
            text = resource.get("code", {}).get("text", "")
            attributes = {
                "clinical_status": resource.get("clinicalStatus", {})
                .get("coding", [{}])[0]
                .get("code", ""),
                "verification_status": resource.get("verificationStatus", {})
                .get("coding", [{}])[0]
                .get("code", ""),
            }
        elif category == "MedicationStatement":
            dosage = resource.get("dosage", [{}])[0]
            text = resource.get("medicationCodeableConcept", {}).get("text", "")
            attributes = {
                "status": resource.get("status", ""),
                "dose": dosage.get("text", ""),
                "route": dosage.get("route", {}).get("text", ""),
                "frequency": dosage.get("timing", {}).get("code", {}).get("text", ""),
            }
        elif category == "Observation":
            text = resource.get("code", {}).get("text", "")
            attributes = {
                "value": resource.get("valueString", ""),
            }
        else:
            text = resource.get("code", {}).get("text", "")
            attributes = {
                "status": resource.get("status", ""),
            }

        segment_value = None
        for extension in resource.get("extension", []):
            if extension.get("url", "").endswith("source-segment-indices"):
                segment_value = extension.get("valueString")
                break

        items.append(
            ExtractionItem(
                category=category,
                text=text,
                segment_indices=tuple(parse_segment_indices(segment_value)),
                attributes=attributes,
                source=f"{step05_path.name}:{resource.get('id', '')}",
            )
        )
    return items


def _load_gold_entries(raw_entries: list[dict[str, Any]], category: str, consultation: str) -> list[ExtractionItem]:
    items: list[ExtractionItem] = []
    for index, raw_entry in enumerate(raw_entries):
        text = raw_entry.get("text", "").strip()
        if not text:
            continue
        segment_indices = tuple(sorted(set(int(value) for value in raw_entry.get("segment_indices", []))))
        attributes = {
            str(key): str(value)
            for key, value in raw_entry.get("attributes", {}).items()
            if str(value).strip()
        }
        items.append(
            ExtractionItem(
                category=category,
                text=text,
                segment_indices=segment_indices,
                attributes=attributes,
                source=f"{consultation}:gold:{category}:{index}",
            )
        )
    return items


def load_gold_items(annotation_path: Path) -> tuple[str, list[ExtractionItem]]:
    """Load gold annotation items from a scaffolded annotation file."""
    payload = read_json(annotation_path)
    consultation = str(payload["consultation"])
    gold_resources = payload.get("gold_resources", {})

    items: list[ExtractionItem] = []
    for category in CATEGORIES:
        items.extend(_load_gold_entries(gold_resources.get(category, []), category, consultation))
    return consultation, items


def consultation_from_annotation_path(annotation_path: Path) -> str:
    """Return the consultation identifier stored in an annotation file."""
    payload = read_json(annotation_path)
    return str(payload["consultation"])


def annotation_status_from_path(annotation_path: Path) -> str:
    """Return the annotation workflow status."""
    payload = read_json(annotation_path)
    return str(payload.get("annotation_status", "draft"))


def evaluate_consultation(
    consultation: str,
    gold_items: list[ExtractionItem],
    predicted_items: list[ExtractionItem],
    *,
    threshold: float,
) -> ConsultationEvaluation:
    """Evaluate one consultation at entity level."""
    errors: list[ErrorRecord] = []
    matched_by_category: dict[str, list[MatchResult]] = {}
    unmatched_gold_by_category: dict[str, list[ExtractionItem]] = {}
    unmatched_predicted_by_category: dict[str, list[ExtractionItem]] = {}
    wrong_attributes_by_category: dict[str, int] = {category: 0 for category in CATEGORIES}

    for category in CATEGORIES:
        category_gold = [item for item in gold_items if item.category == category]
        category_predicted = [item for item in predicted_items if item.category == category]
        same_category_matches, unmatched_gold, unmatched_predicted = greedy_match(
            category_gold,
            category_predicted,
            threshold=threshold,
        )

        for match in same_category_matches:
            mismatch_count = attribute_mismatch_count(match.gold, match.predicted)
            wrong_attributes_by_category[category] += mismatch_count
            if mismatch_count:
                errors.append(
                    ErrorRecord(
                        consultation=consultation,
                        error_type="wrong_attributes",
                        category=category,
                        gold_text=match.gold.text,
                        predicted_text=match.predicted.text,
                        score=match.score,
                    )
                )

        matched_by_category[category] = same_category_matches
        unmatched_gold_by_category[category] = unmatched_gold
        unmatched_predicted_by_category[category] = unmatched_predicted

    cross_gold = [item for category in CATEGORIES for item in unmatched_gold_by_category[category]]
    cross_predicted = [item for category in CATEGORIES for item in unmatched_predicted_by_category[category]]
    cross_matches, remaining_gold, remaining_predicted = greedy_match(
        cross_gold,
        cross_predicted,
        threshold=threshold,
    )

    wrong_category_by_category: dict[str, int] = {category: 0 for category in CATEGORIES}
    for match in cross_matches:
        if match.gold.category == match.predicted.category:
            continue
        wrong_category_by_category[match.gold.category] += 1
        errors.append(
            ErrorRecord(
                consultation=consultation,
                error_type="wrong_category",
                category=match.gold.category,
                gold_text=match.gold.text,
                predicted_text=match.predicted.text,
                score=match.score,
            )
        )

    remaining_gold_by_category: dict[str, list[ExtractionItem]] = {category: [] for category in CATEGORIES}
    remaining_predicted_by_category: dict[str, list[ExtractionItem]] = {category: [] for category in CATEGORIES}
    for item in remaining_gold:
        remaining_gold_by_category[item.category].append(item)
    for item in remaining_predicted:
        remaining_predicted_by_category[item.category].append(item)

    summaries: dict[str, CategorySummary] = {}
    overall_gold_count = 0
    overall_predicted_count = 0
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_wrong_category = 0
    overall_wrong_attributes = 0
    overall_hallucinated = 0
    overall_missed = 0

    for category in CATEGORIES:
        category_gold = [item for item in gold_items if item.category == category]
        category_predicted = [item for item in predicted_items if item.category == category]
        tp = len(matched_by_category[category])
        fp = len(category_predicted) - tp
        fn = len(category_gold) - tp
        hallucinated = len(remaining_predicted_by_category[category])
        missed = len(remaining_gold_by_category[category])

        for item in remaining_predicted_by_category[category]:
            errors.append(
                ErrorRecord(
                    consultation=consultation,
                    error_type="hallucinated",
                    category=category,
                    gold_text="",
                    predicted_text=item.text,
                    score=0.0,
                )
            )
        for item in remaining_gold_by_category[category]:
            errors.append(
                ErrorRecord(
                    consultation=consultation,
                    error_type="missed",
                    category=category,
                    gold_text=item.text,
                    predicted_text="",
                    score=0.0,
                )
            )

        precision, recall, f1 = metric_triplet(tp, fp, fn)
        summaries[category] = CategorySummary(
            gold_count=len(category_gold),
            predicted_count=len(category_predicted),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            wrong_category=wrong_category_by_category[category],
            wrong_attributes=wrong_attributes_by_category[category],
            hallucinated=hallucinated,
            missed=missed,
        )

        overall_gold_count += len(category_gold)
        overall_predicted_count += len(category_predicted)
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        overall_wrong_category += wrong_category_by_category[category]
        overall_wrong_attributes += wrong_attributes_by_category[category]
        overall_hallucinated += hallucinated
        overall_missed += missed

    overall_precision, overall_recall, overall_f1 = metric_triplet(overall_tp, overall_fp, overall_fn)
    overall = CategorySummary(
        gold_count=overall_gold_count,
        predicted_count=overall_predicted_count,
        tp=overall_tp,
        fp=overall_fp,
        fn=overall_fn,
        precision=overall_precision,
        recall=overall_recall,
        f1=overall_f1,
        wrong_category=overall_wrong_category,
        wrong_attributes=overall_wrong_attributes,
        hallucinated=overall_hallucinated,
        missed=overall_missed,
    )
    return ConsultationEvaluation(
        consultation=consultation,
        categories=summaries,
        overall=overall,
        errors=errors,
    )


def predicted_candidates_for_annotation(step05_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Export prediction candidates into the gold scaffold shape."""
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for item in extract_predicted_items(step05_path):
        grouped[item.category].append(
            {
                "text": item.text,
                "segment_indices": list(item.segment_indices),
                "attributes": item.attributes,
            }
        )
    return grouped


def transcript_markdown(step04_path: Path, consultation: str) -> str:
    """Render a readable transcript packet for manual annotation."""
    payload = read_json(step04_path)
    lines = [
        f"# {consultation} Transcript Packet",
        "",
        "Use this file to annotate `gold_resources` in the paired JSON scaffold.",
        "Only annotate explicit clinical content for Condition, MedicationStatement, Observation, and Procedure.",
        "",
        "## Segments",
        "",
    ]
    for index, segment in enumerate(payload["segments"]):
        cleaned = str(segment.get("cleaned_text", "")).strip()
        if not cleaned:
            continue
        lines.append(
            f"- [{index}] {segment.get('speaker_role', 'UNKNOWN')} "
            f"{segment.get('start', 0.0):.2f}s-{segment.get('end', 0.0):.2f}s: {cleaned}"
        )
    lines.append("")
    return "\n".join(lines)
