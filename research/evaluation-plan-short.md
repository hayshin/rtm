# Short Evaluation Plan for Fast Implementation

This is a reduced evaluation plan designed for limited time and current repo reality.

Right now the project has:
- pipeline outputs for a subset of PriMock57 consultations
- PriMock57 reference transcripts in TextGrid format
- no current evaluation harness
- no current gold extraction or gold FHIR annotations

Because of that, the first defensible milestone is to benchmark the existing pipeline on audio and transcript quality before promising downstream semantic evaluation.

## 1. Evaluation Strategy

This short plan uses a phased rollout:

### Phase 1. Audio -> Transcript Benchmarking

Evaluate the current pipeline against PriMock57 reference transcripts.

Required metrics:
- `WER`
- optional `CER`

Comparison targets:
- `step03` raw ASR transcript
- optional `step04` cleaned transcript as an ablation

Why first:
- reference data already exists
- pipeline outputs already exist
- no new annotation is required
- this produces the first real benchmark numbers for the project

### Phase 1b. Speaker Roles If Feasible

Evaluate speaker-role attribution only if it can be derived reliably from existing PriMock57 transcript timings.

Metrics:
- segment-level speaker-role accuracy
- confusion matrix for `PHYSICIAN` vs `PATIENT`

Reference:
- PriMock57 doctor and patient TextGrid intervals

Fallback:
- if alignment is not reliable enough, defer role evaluation to the same manually annotated subset used later for extraction

### Phase 2. Transcript -> Clinical Extraction

Evaluate extraction only after creating a small annotated subset.

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

Reference:
- manual gold annotations on 5 consultations minimum

Why deferred:
- no extraction gold set exists yet
- extraction evaluation is the strongest downstream metric, but it requires new annotation work

### Phase 3. FHIR Output

Evaluate structural validity on all processed consultations after transcript benchmarking is in place.

Metrics:
- FHIR schema validation pass rate
- number of validation errors per consultation

Why keep it:
- it is cheap to compute
- it supports the interoperability claim
- it does not require gold FHIR bundles

## 2. Core Claim

Recommended framing:

> PriMock57 provides the benchmark source for clinical audio and transcript-level evaluation, but there is no established protocol for end-to-end audio-to-FHIR assessment. This study therefore begins by benchmarking transcription quality on PriMock57, adds speaker-role evaluation where existing labels support it, and defers extraction-level semantic evaluation to a small manually annotated subset.

## 3. Phase 1 Benchmark Package

The minimum benchmark package to implement now is:

1. export reference transcripts from PriMock57 TextGrids
2. export predicted transcripts from pipeline outputs
3. compute `WER` on all currently processed consultations
4. compute optional `CER`
5. compute speaker-role accuracy only if role alignment from TextGrids is reliable

Expected artifacts:
- reference transcript export
- predicted transcript export
- benchmark results table
- optional speaker-role comparison table

## 4. Metrics to Implement Now

### A. Audio -> Transcript

Evaluate on all consultations already processed in `batch_outputs/primock57_pipeline/`.

Required metric:
- `WER`

Optional metrics:
- `CER`
- comparison of `step03` vs `step04`

Reference:
- PriMock57 doctor and patient TextGrid transcripts merged in timestamp order

Recommended prediction source of truth:
- use `step03` for the primary ASR benchmark
- use `step04` only as a cleanup ablation

### B. Speaker Roles

Evaluate only if role labels can be matched from existing PriMock57 timing information.

Metrics:
- segment-level role accuracy
- confusion matrix

Prediction source:
- `step04.speaker_role`

Reference source:
- doctor and patient TextGrid intervals

## 5. What Not to Promise Yet

Do not claim:
- that the project already has a full benchmark suite
- full end-to-end semantic correctness
- gold-standard FHIR bundle fidelity
- extraction precision / recall / F1 before the gold subset exists

Claim instead:
- the project now has a reproducible transcript benchmark
- audio-first evaluation is the first completed milestone
- downstream extraction and FHIR semantic evaluation are planned separately

## 6. Minimal Annotation Plan for Later Phases

After Phase 1 is complete, annotate only what is needed for fast extraction F1.

Recommended subset:
- 5 consultations minimum
- 10 consultations if feasible

For each consultation annotate:
- speaker role by utterance or segment if still needed
- spans for conditions
- spans for medications
- spans for observations
- spans for procedures

Optional if time allows:
- negation
- dosage
- frequency
- temporal expressions

This avoids the cost of building full gold FHIR resources at the start.

## 7. Fast Baselines and Ablations

### Baseline 1. `step03` Raw ASR

Purpose:
- primary benchmark for current transcription quality

### Baseline 2. `step04` Cleaned Transcript

Purpose:
- measures whether cleanup improves transcript quality relative to reference text

This is useful as a lightweight ablation, not the primary benchmark.

### Baseline 3. Gold Transcript -> Extraction -> FHIR

Purpose:
- later baseline for separating ASR error from extraction error

This is valuable, but it belongs after Phase 1 benchmarking is in place.

## 8. Minimum Results Tables

### Table A. ASR Benchmark

Rows:
- one row per consultation
- one summary row

Columns:
- consultation id
- transcript source
- WER
- optional CER

### Table B. Cleanup Comparison

Rows:
- `step03`
- `step04`

Columns:
- WER
- CER

### Table C. Speaker-Role Evaluation

Include only if feasible in Phase 1b.

Rows:
- overall
- `PHYSICIAN`
- `PATIENT`

Columns:
- accuracy or support
- confusion counts

## 9. Implementation Order

Implement in this order:

1. parse PriMock57 TextGrid transcripts into merged reference transcripts
2. read pipeline `step03` outputs and compute `WER`
3. add optional `CER`
4. compare `step04` cleaned transcripts as an ablation
5. compute speaker-role accuracy from TextGrid overlaps if reliable
6. only then define the lightweight annotation format for extraction

## 10. Shortest Defensible Package

The shortest defensible evaluation package is:
- `WER` on all currently processed consultations
- optional `CER`
- optional `step03` vs `step04` comparison
- optional speaker-role accuracy if existing PriMock57 labels support it

This is realistic, publishable as a feasibility-focused benchmark, and does not require new gold extraction or FHIR annotations before producing results.
