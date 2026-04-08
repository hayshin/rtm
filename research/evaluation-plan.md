# Evaluation Plan for "Designing an Automated Pipeline for Mapping Unstructured Clinical Audio to HL7 FHIR-Compliant Data Structures"

This note lists concrete benchmarks, evaluations, and baselines that can be added to strengthen the paper.

## 1. Benchmark Framing

The paper should not claim that there is no benchmark at all.

A more accurate claim is:
- `PriMock57` is the benchmark dataset for the clinical-audio part of the task.
- There is no widely accepted `end-to-end audio-to-FHIR benchmark protocol`.
- Therefore, this study should define an evaluation protocol on top of PriMock57.

Recommended wording:

> PriMock57 provides a public benchmark for primary-care consultation audio and transcript-level evaluation, but no established benchmark protocol currently exists for evaluating the full pathway from raw clinical audio to HL7 FHIR-compliant structured output. This study therefore defines an end-to-end evaluation protocol spanning transcription quality, speaker-role attribution, structural FHIR validity, and resource-level semantic fidelity.

## 2. Main Evaluation Layers

The pipeline should be evaluated at four levels.

### A. Audio -> Transcript

Goal:
- Measure how accurately the pipeline transcribes clinical speech.

Metrics:
- `WER` (Word Error Rate)
- `CER` (Character Error Rate), optional
- `Medical Concept WER` or concept-sensitive transcription accuracy

Benchmark source:
- PriMock57 reference transcripts

Why it matters:
- If ASR quality is weak, every downstream stage is affected.

### B. Transcript -> Speaker Roles

Goal:
- Measure how correctly the system assigns `physician` and `patient` roles.

Metrics:
- speaker-role accuracy
- confusion matrix for `physician` vs `patient`
- `WDER` if word-level diarization evaluation is possible

Benchmark source:
- PriMock57 utterance or speaker labels, if available
- otherwise manual annotation for a subset

Why it matters:
- Role attribution errors directly affect clinical meaning.

### C. Transcript -> Clinical Extraction

Goal:
- Measure whether symptoms, diagnoses, medications, observations, and procedures are extracted correctly.

Metrics:
- precision
- recall
- F1
- per-resource-type F1
- attribute-level accuracy for key fields

Suggested resource groups:
- `Condition`
- `MedicationStatement`
- `Observation`
- `Procedure`
- `Encounter`

Important attributes to score:
- mention present or absent
- code or normalized label
- negation / assertion status
- temporal context if available
- dosage, route, frequency for medications

Benchmark source:
- manual gold annotations on a subset of consultations
- consultation notes as weak reference if full gold FHIR is not feasible

Why it matters:
- This is the main semantic evaluation missing from the current paper.

### D. Clinical Extraction -> FHIR Output

Goal:
- Measure both structural correctness and semantic faithfulness of final bundles.

Metrics:
- FHIR schema validation pass rate
- number of validation errors per consultation
- profile conformance pass rate, if HAPI validator is added
- provenance coverage rate
- semantic bundle accuracy on a gold subset

Why it matters:
- Structural validity alone is not enough; it must be paired with semantic correctness.

## 3. Baselines to Add

At least one baseline is necessary. Ideally add three.

### Baseline 1. Raw ASR -> Extraction -> FHIR

Pipeline:
- diarization
- ASR
- direct extraction to FHIR
- no LLM transcript cleanup

Purpose:
- tests whether Step 4 actually improves downstream extraction

Expected comparison:
- lower transcript quality
- lower resource precision/recall
- possibly similar schema validity if final assembly is still constrained

### Baseline 2. Gold Transcript -> Extraction -> FHIR

Pipeline:
- human reference transcript instead of ASR output
- same extraction and FHIR generation stages

Purpose:
- isolates the effect of upstream speech errors
- gives an approximate upper bound for the downstream extraction stage

Expected comparison:
- better extraction metrics than raw-audio pipeline
- helps distinguish ASR limitations from FHIR mapping limitations

### Baseline 3. Simple Rule-Based or Minimal Prompt Baseline

Possible forms:
- direct single-pass prompt from transcript to structured JSON
- simple keyword/rule extraction for medications and conditions
- transcript -> note only, without FHIR mapping

Purpose:
- shows that the proposed modular pipeline is better than a naive approach

Expected comparison:
- lower semantic fidelity
- more hallucinations or poorer resource structure

## 4. Ablation Studies to Add

Ablations are often easier than collecting a completely new baseline.

### Ablation A. Without transcript cleanup

Compare:
- full pipeline
- same pipeline without Step 4 cleanup

Measure:
- WER after cleanup vs before cleanup
- extraction F1 difference
- change in role assignment accuracy

### Ablation B. Without ontology normalization

Compare:
- full pipeline
- extraction without SNOMED / RxNorm / LOINC normalization

Measure:
- code-level accuracy
- interoperability score
- validation pass rate if coding affects required fields

### Ablation C. Without provenance construction

Compare:
- full pipeline
- same output without provenance

Measure:
- provenance coverage
- auditability, not semantic quality

This is weaker scientifically, but useful if auditability is part of your contribution.

## 5. Minimum Defensible Evaluation if Time Is Limited

If a full gold-standard evaluation is too expensive, the minimum defensible package is:

1. `WER` on all PriMock57 Day 1 consultations
2. speaker-role accuracy on a manually annotated subset of 5 to 10 consultations
3. resource-level precision/recall/F1 on a manually annotated subset of 5 consultations
4. schema validation pass rate on all processed consultations
5. one ablation: with and without Step 4 transcript cleanup

This would already make the paper much stronger than its current form.

## 6. Recommended Gold Annotation Strategy

If full FHIR annotation is too much work, annotate a reduced target.

Recommended subset:
- 5 to 10 consultations

Annotate only these resource types:
- `Condition`
- `MedicationStatement`
- `Observation`
- `Procedure`

For each resource, annotate:
- mention span in transcript
- normalized label or code
- speaker source
- negated / present
- key attributes

This subset is enough to estimate precision/recall and discuss error categories.

## 7. Tables the Paper Can Add

### Table A. Benchmark and Evaluation Protocol

Columns:
- pipeline stage
- task
- dataset / reference
- metric
- output artifact

### Table B. Baseline Comparison

Rows:
- raw ASR -> FHIR
- gold transcript -> FHIR
- proposed pipeline

Columns:
- WER
- role accuracy
- extraction F1
- validation pass rate

### Table C. Per-Resource Evaluation

Rows:
- Condition
- MedicationStatement
- Observation
- Procedure

Columns:
- precision
- recall
- F1
- common error types

## 8. How to Reframe the Current Paper if Metrics Are Not Ready Yet

If you cannot compute the missing metrics before submission, reframe the contribution honestly.

Do not claim:
- full evaluation of clinical meaningfulness
- strong semantic correctness
- production readiness

Claim instead:
- end-to-end feasibility
- structural interoperability
- reproducible pipeline design
- benchmarked execution on PriMock57 Day 1
- identified semantic failure modes that motivate future evaluation

Recommended wording:

> The present study should be interpreted as a feasibility-oriented benchmarked systems paper rather than a complete semantic accuracy study. Its strongest evidence concerns end-to-end execution and FHIR structural validity on PriMock57; quantitative assessment of semantic fidelity remains future work.

## 9. Priority Order

If only a few additions are possible, do them in this order:

1. Add `WER` against PriMock57 reference transcripts
2. Add one baseline: `gold transcript -> extraction -> FHIR`
3. Add manual precision/recall on a small annotated subset
4. Add speaker-role accuracy
5. Add ablation for Step 4 transcript cleanup

## 10. Short Summary

The strongest version of the paper would evaluate:
- benchmark dataset: `PriMock57`
- baselines: `raw ASR`, `gold transcript`, and a naive extraction baseline
- metrics: `WER`, role accuracy, precision/recall/F1, and FHIR validation pass rate
- ablations: especially whether transcript cleanup improves downstream FHIR extraction

Without these additions, the work is best framed as a structurally validated feasibility pipeline, not a fully evaluated end-to-end clinical NLP system.
