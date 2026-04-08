# Short Evaluation Plan for Fast Implementation

This is a reduced evaluation plan designed for limited time.

It keeps the strongest metrics that can be implemented quickly and avoids any requirement to create full gold-standard FHIR resources.

## 1. Evaluation Scope

This short plan evaluates the pipeline at four practical levels:
- audio transcription quality
- speaker-role attribution
- clinical information extraction
- FHIR structural validity

It does not attempt full semantic evaluation of generated FHIR bundles.

## 2. Core Claim

Recommended framing:

> PriMock57 provides a benchmark for clinical audio and transcript-level evaluation, but there is no standard protocol for end-to-end audio-to-FHIR assessment. Given time and annotation constraints, this study evaluates the pipeline through transcription quality, speaker-role attribution, extraction accuracy on a small manually annotated subset, and FHIR structural validity.

## 3. Metrics to Implement

### A. Audio -> Transcript

Evaluate on all available PriMock57 consultations.

Metrics:
- `WER`
- optional `CER`

Reference:
- PriMock57 reference transcripts

Why keep it:
- easy to compute
- necessary for grounding all downstream results

### B. Transcript -> Speaker Roles

Evaluate on a small manually annotated subset.

Metrics:
- speaker-role accuracy
- confusion matrix for `physician` vs `patient`

Reference:
- PriMock57 labels if available
- otherwise manual annotation for 5 to 10 consultations

Why keep it:
- directly relevant to dialogue meaning
- still feasible with a small subset

### C. Transcript -> Clinical Extraction

Evaluate on a small manually annotated subset.

Metrics:
- precision
- recall
- F1
- per-category F1

Recommended categories:
- `Condition`
- `Medication`
- `Observation`
- `Procedure`

Optional attributes if feasible:
- negation
- dosage
- frequency
- temporal mention

Reference:
- manual gold annotations on 5 consultations minimum

Why keep it:
- this is the strongest semantic evaluation you can produce quickly

### D. FHIR Output

Evaluate on all processed consultations.

Metrics:
- FHIR schema validation pass rate
- number of validation errors per consultation

Why keep it:
- this is fast to implement
- it supports the interoperability part of the paper even without gold FHIR bundles

## 4. What Not to Promise

Do not claim:
- full end-to-end semantic correctness
- gold-standard FHIR bundle fidelity
- complete clinical validity of the structured output

Claim instead:
- feasibility of the pipeline
- benchmarked transcription and extraction performance
- structurally valid FHIR generation

## 5. Minimal Annotation Plan

Annotate only what is needed for fast F1 evaluation.

Recommended subset:
- 5 consultations minimum
- 10 consultations if feasible

For each consultation annotate:
- speaker role by utterance or segment
- spans for conditions
- spans for medications
- spans for observations
- spans for procedures

If time allows, also annotate:
- negation
- dosage and frequency for medications
- temporal expressions

This avoids the cost of building full gold FHIR resources.

## 6. Fast Baselines

Use only baselines that are cheap and informative.

### Baseline 1. Raw ASR -> Extraction -> FHIR

Pipeline:
- diarization
- ASR
- extraction
- FHIR generation
- no transcript cleanup

Purpose:
- shows whether cleanup improves results

### Baseline 2. Gold Transcript -> Extraction -> FHIR

Pipeline:
- reference transcript
- same extraction and FHIR generation stages

Purpose:
- separates ASR error from extraction error

This is the most valuable baseline if you only add one.

## 7. Fast Ablation

### Ablation. Without transcript cleanup

Compare:
- full pipeline
- same pipeline without cleanup

Measure:
- WER difference
- extraction precision / recall / F1 difference
- speaker-role accuracy difference if relevant

This is the cheapest meaningful ablation.

## 8. Minimum Results Tables

### Table A. Main Evaluation

Rows:
- transcription
- speaker roles
- extraction
- FHIR validation

Columns:
- dataset or subset
- metric
- score

### Table B. Per-Category Extraction

Rows:
- Condition
- Medication
- Observation
- Procedure

Columns:
- precision
- recall
- F1

### Table C. Baseline or Ablation Comparison

Rows:
- full pipeline
- without cleanup
- gold transcript pipeline

Columns:
- WER
- role accuracy
- extraction F1
- validation pass rate

## 9. Implementation Order

If we implement this together quickly, the best order is:

1. compute `WER` from PriMock57 references
2. define a lightweight annotation format for 5 consultations
3. evaluate extraction precision / recall / F1
4. evaluate speaker-role accuracy
5. run FHIR validation and report pass rate
6. compare full pipeline vs no-cleanup pipeline

## 10. Short Summary

The shortest defensible evaluation package is:
- `WER` on PriMock57
- speaker-role accuracy on a small subset
- extraction precision / recall / F1 on a small subset
- FHIR validation pass rate on all outputs
- one comparison between full pipeline and no-cleanup pipeline

This is realistic, publishable as a feasibility-focused evaluation, and does not require full gold FHIR annotations.
