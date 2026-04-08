# Short Evaluation Plan for Fast Implementation

This is a reduced evaluation plan designed for limited time and current repo reality.

Right now the project has:
- pipeline outputs for a subset of PriMock57 consultations
- PriMock57 reference transcripts in TextGrid format
- a working audio evaluation harness
- no current gold extraction or gold FHIR annotations

The first defensible milestone was to benchmark the existing pipeline on audio and transcript quality before promising downstream semantic evaluation. That milestone is now complete for the 15 processed Day~1 consultations.

## 1. Evaluation Strategy

This short plan uses a phased rollout:

### Phase 1. Audio -> Transcript Benchmarking

Status: done.

The current pipeline has been evaluated against PriMock57 reference transcripts for the 15 processed Day~1 consultations.

Implemented metrics:
- `WER`
- optional `CER` remains available but is not part of the default fast run

Implemented comparison targets:
- `step03` raw ASR transcript
- `step04` cleaned transcript as an ablation

Current outcome:
- reference data already exists
- pipeline outputs already exist
- no new annotation is required
- this produced the first real benchmark numbers for the project
- `step03` corpus-level `WER = 0.3224`
- `step04` corpus-level `WER = 0.5946`

### Phase 1b. Speaker Roles If Feasible

Status: done for the current Day~1 batch via overlap with existing PriMock57 transcript timings.

Metrics:
- segment-level speaker-role accuracy
- confusion matrix for `PHYSICIAN` vs `PATIENT`

Reference:
- PriMock57 doctor and patient TextGrid intervals

Fallback:
- if alignment is not reliable enough, defer role evaluation to the same manually annotated subset used later for extraction

Current outcome:
- `step04` speaker-role accuracy = `0.8963` over 2651 scored segments

### Phase 2. Transcript -> Clinical Extraction

Status: pending.

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

Status: structurally done, semantically pending.

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

Done:

1. export reference transcripts from PriMock57 TextGrids
2. export predicted transcripts from pipeline outputs
3. compute `WER` on all currently processed consultations
4. compare `step03` and `step04`
5. compute speaker-role accuracy from TextGrid overlap

Still optional:

6. compute `CER`

Expected artifacts:
- reference transcript export
- predicted transcript export
- benchmark results table
- optional speaker-role comparison table

## 4. Metrics Implemented Now

### A. Audio -> Transcript

Evaluated on all consultations currently processed in `batch_outputs/primock57_pipeline/`.

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
This has been implemented for the current Day~1 subset using interval overlap.

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

## 9. Next Implementation Order

Phase 1 is complete. Next:

1. define the lightweight annotation format for extraction
2. annotate 5 consultations minimum
3. evaluate extraction precision / recall / F1
4. extend structural FHIR reporting if needed
5. add optional `CER` or concept-sensitive transcript metrics later

## 10. Shortest Defensible Package

The shortest defensible evaluation package is:
- `WER` on all currently processed consultations
- `step03` vs `step04` comparison
- speaker-role accuracy from existing PriMock57 labels
- optional `CER`

This package is now implemented for the processed Day~1 subset. The next missing piece is extraction evaluation on a small annotated gold subset.
