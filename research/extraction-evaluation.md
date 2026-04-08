# Extraction Evaluation Workflow

This adds the missing Phase 2 evaluation layer: manual gold annotation for Step 5 extraction outputs.

## What exists now

- `scripts/scaffold_extraction_gold.py`
  - builds annotation packets from existing `step04` and `step05` outputs
  - defaults to the first 5 processed consultations
- `scripts/evaluate_extraction_gold.py`
  - scores Step 5 outputs against manual gold annotations
  - reports precision, recall, F1, per-category metrics, and error buckets
- `annotations/extraction_gold/`
  - intended home for manual annotation packets

## Gold annotation format

Each consultation JSON contains:

- `consultation`
- `annotation_status`
- `annotator`
- `notes`
- `transcript_packet`
- `gold_resources`
- `predicted_candidates`

Only `gold_resources` is scored.

Each gold item should look like:

```json
{
  "text": "Diarrhoea",
  "segment_indices": [8, 14, 21],
  "attributes": {
    "verification_status": "confirmed"
  }
}
```

Supported categories:

- `Condition`
- `MedicationStatement`
- `Observation`
- `Procedure`

Supported optional attributes:

- `Condition`: `clinical_status`, `verification_status`
- `MedicationStatement`: `status`, `dose`, `route`, `frequency`
- `Observation`: `value`
- `Procedure`: `status`

## Recommended workflow

1. Create or refresh packets:

```bash
PYTHONPATH=src uv run python scripts/scaffold_extraction_gold.py --limit 5
```

2. Annotate the generated JSON files in `annotations/extraction_gold/`, using the paired transcript markdown files.

3. Run the evaluator:

```bash
PYTHONPATH=src uv run python scripts/evaluate_extraction_gold.py \
  --output-csv outputs/extraction_eval_metrics.csv \
  --error-report outputs/extraction_eval_errors.json
```

## Scoring notes

- Entity matching is approximate, using normalized text overlap plus segment-index overlap.
- Default match threshold is `0.65`.
- Main metrics are entity-level precision, recall, and F1.
- Error buckets are:
  - `wrong_category`
  - `wrong_attributes`
  - `hallucinated`
  - `missed`

This is a fast semantic evaluation layer for the paper. It is not yet full gold-FHIR fidelity scoring.
