# Evaluation Plan for "Designing an Automated Pipeline for Mapping Unstructured Clinical Audio to HL7 FHIR-Compliant Data Structures"

This note lists concrete benchmarks, evaluations, baselines, and ablations that can strengthen the paper.

The plan is informed by three useful patterns from prior work:
- admission-note to EHR studies usually evaluate extraction well, but often leave terminology normalization and final FHIR mapping under-evaluated
- FHIR normalization studies show that individual FHIR elements can be evaluated directly with gold FHIR-based annotations
- end-to-end semantic pipeline papers are strongest when they combine extraction metrics with semantic completeness, interoperability, baselines, and ablations

## 1. Benchmark Framing

The paper should not claim that there is no benchmark at all.

A more accurate claim is:
- `PriMock57` is the benchmark dataset for the clinical-audio part of the task
- there is no widely accepted `end-to-end audio-to-FHIR benchmark protocol`
- therefore, this study should define an evaluation protocol on top of PriMock57

Recommended wording:

> PriMock57 provides a public benchmark for primary-care consultation audio and transcript-level evaluation, but no established benchmark protocol currently exists for evaluating the full pathway from raw clinical audio to HL7 FHIR-compliant structured output. This study therefore defines an end-to-end evaluation protocol spanning transcription quality, speaker-role attribution, clinical extraction, terminology normalization, and FHIR-level semantic fidelity.

## 2. Main Evaluation Layers

The pipeline should be evaluated at six levels.

### A. Audio -> Transcript

Goal:
- measure how accurately the pipeline transcribes clinical speech

Metrics:
- `WER` (Word Error Rate)
- `CER` (Character Error Rate), optional
- `Medical Concept WER` or concept-sensitive transcription accuracy

Benchmark source:
- PriMock57 reference transcripts

Why it matters:
- if ASR quality is weak, every downstream stage is affected

### B. Transcript -> Speaker Roles

Goal:
- measure how correctly the system assigns `physician` and `patient` roles

Metrics:
- speaker-role accuracy
- confusion matrix for `physician` vs `patient`
- `WDER` if word-level diarization evaluation is possible

Benchmark source:
- PriMock57 utterance or speaker labels, if available
- otherwise manual annotation for a subset

Why it matters:
- role attribution errors directly affect clinical meaning

### C. Transcript -> Clinical Extraction

Goal:
- measure whether symptoms, diagnoses, medications, observations, and procedures are extracted correctly

Metrics:
- precision
- recall
- F1
- entity-level F1
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
- normalized label
- negation / assertion status
- temporal context if available
- dosage, route, frequency for medications
- observation value and unit where applicable
- speaker source

Benchmark source:
- manual gold annotations on a subset of consultations
- consultation notes as weak reference if full gold FHIR is not feasible

Why it matters:
- this is the main semantic evaluation missing from the current paper

### D. Extraction -> Relation and Attribute Assembly

Goal:
- measure whether the pipeline correctly links extracted concepts to the attributes and relations needed for valid FHIR resources

Metrics:
- precision
- recall
- F1 for relation or attachment extraction
- dosage-to-medication attachment accuracy
- observation-to-value attachment accuracy
- temporal attachment accuracy
- assertion linkage accuracy

Examples of links to score:
- medication -> dosage
- observation -> value
- condition -> negation / assertion
- event -> time
- mention -> speaker role

Benchmark source:
- manual gold annotation on the same subset used for extraction evaluation

Why it matters:
- many FHIR errors are not span-detection errors; they are assembly errors

### E. Extraction -> Normalization

Goal:
- measure whether extracted mentions are normalized to the correct standard terminology concepts

Metrics:
- exact code match accuracy
- normalized label match accuracy
- precision / recall / F1 at the code level
- top-k candidate hit rate, optional

Recommended code systems:
- `SNOMED CT` for conditions and findings
- `RxNorm` for medications
- `LOINC` for observations where appropriate

Benchmark source:
- gold normalized codes on the annotated subset

Why it matters:
- prior pipeline papers often stop short of quantitative normalization evaluation; this study should not repeat that gap

### F. Clinical Extraction -> FHIR Output

Goal:
- measure both structural correctness and semantic faithfulness of final bundles

Metrics:
- FHIR schema validation pass rate
- number of validation errors per consultation
- profile conformance pass rate, if HAPI validator is added
- provenance coverage rate
- semantic completeness
- interoperability or bundle similarity score
- semantic bundle accuracy on a gold subset
- element-level precision / recall / F1 for key FHIR fields

Recommended key FHIR elements to score separately:
- `Condition.code`
- `Condition.clinicalStatus`
- `Condition.verificationStatus`
- `MedicationStatement.medicationCodeableConcept`
- `MedicationStatement.dosage`
- `Observation.code`
- `Observation.value[x]`
- `Observation.effective[x]`
- `Procedure.code`
- `Encounter.reasonCode` or equivalent target field if used

Suggested end-to-end definitions:
- `semantic completeness`: proportion of gold-required fields that are correctly populated in the predicted FHIR output
- `interoperability score`: structural and semantic similarity between predicted bundles and gold or reference bundles

Why it matters:
- structural validity alone is not enough; it must be paired with semantic correctness

## 3. Gold Annotation Strategy

The strongest implementation pattern is to create a small gold-standard subset that supports transcript, extraction, normalization, and FHIR-level evaluation at once.

### Recommended subset size

- 5 to 10 consultations for a minimum defensible study
- 10 to 20 consultations if resources allow

### Recommended annotation unit

Annotate both:
- mention-level information in transcripts
- key FHIR element values derived from those mentions

### Recommended resource scope

If full FHIR annotation is too expensive, annotate only:
- `Condition`
- `MedicationStatement`
- `Observation`
- `Procedure`

Add `Encounter` only if the pipeline meaningfully populates it.

### For each annotated item, capture

- mention span in transcript
- normalized label
- gold terminology code
- speaker source
- negated / present / hypothetical if relevant
- temporal context
- relation links needed for assembly
- target FHIR resource type
- target FHIR element values for core fields

### Annotation process

- start from PriMock57 reference transcripts
- create reduced gold FHIR-style annotations for the subset
- if possible, have a second reviewer validate at least a portion of the subset

Why this matters:
- this follows the strongest pattern from FHIR normalization work, where FHIR-based gold annotations enable direct element-level evaluation instead of only high-level bundle checks

## 4. Baselines to Add

At least one baseline is necessary. Ideally add four or five.

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
- better extraction metrics than the raw-audio pipeline
- helps distinguish ASR limitations from FHIR mapping limitations

### Baseline 3. Simple Rule-Based or Minimal Prompt Baseline

Possible forms:
- direct single-pass prompt from transcript to structured JSON
- simple keyword or rule extraction for medications and conditions
- transcript -> note only, without full FHIR mapping

Purpose:
- shows that the proposed modular pipeline is better than a naive approach

Expected comparison:
- lower semantic fidelity
- more hallucinations or poorer resource structure

### Baseline 4. No-Normalization Baseline

Pipeline:
- extraction from transcript
- resource assembly with text labels only
- no ontology linking

Purpose:
- isolates the contribution of terminology normalization to interoperability

Expected comparison:
- similar mention detection
- worse code-level accuracy
- worse semantic completeness and interoperability

### Baseline 5. No-Relation or Flat-Assembly Baseline

Pipeline:
- extract mentions only
- map each mention independently to FHIR
- do not explicitly recover dosage, value, temporal, or assertion links

Purpose:
- tests whether structured relation and attribute assembly improves realistic FHIR population

Expected comparison:
- weaker medication dosage population
- weaker observation value population
- lower semantic completeness even if schema validity remains acceptable

## 5. Ablation Studies to Add

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
- semantic completeness
- interoperability score
- validation pass rate if coding affects required fields

### Ablation C. Without relation or attribute assembly

Compare:
- full pipeline
- same pipeline without relation and attribute assembly

Measure:
- relation F1
- medication dosage accuracy
- observation value accuracy
- semantic completeness
- interoperability score

### Ablation D. Without FHIR validation

Compare:
- full pipeline
- same pipeline without validation step

Measure:
- validation pass rate
- mean number of validation errors
- semantic completeness
- interoperability score

### Ablation E. Without provenance construction

Compare:
- full pipeline
- same output without provenance

Measure:
- provenance coverage
- auditability, not semantic quality

This is weaker scientifically, but useful if auditability is part of the contribution.

## 6. Recommended Metrics Tables

The paper should make explicit which metric belongs to which stage.

### Table A. Stage-Level Evaluation Protocol

Columns:
- pipeline stage
- evaluation unit
- dataset / reference
- metric
- rationale

Recommended rows:
- audio transcription
- speaker-role attribution
- clinical extraction
- relation / attribute assembly
- terminology normalization
- FHIR validity and semantics

### Table B. End-to-End Baseline Comparison

Rows:
- raw ASR -> FHIR
- gold transcript -> FHIR
- naive or rule-based baseline
- no-normalization baseline
- proposed pipeline

Columns:
- WER
- role accuracy
- extraction F1
- relation F1
- code accuracy
- semantic completeness
- interoperability score
- validation pass rate

### Table C. Per-Resource / Per-Element Evaluation

Rows:
- Condition
- MedicationStatement
- Observation
- Procedure

Columns:
- mention F1
- code accuracy
- attribute accuracy
- element-level F1
- common error types

## 7. Minimum Defensible Evaluation if Time Is Limited

If a full gold-standard evaluation is too expensive, the minimum defensible package is:

1. `WER` on all PriMock57 Day 1 consultations
2. speaker-role accuracy on a manually annotated subset of 5 to 10 consultations
3. resource-level precision/recall/F1 on a manually annotated subset of 5 consultations
4. code-level normalization accuracy on the same subset
5. schema validation pass rate on all processed consultations
6. one ablation: with and without Step 4 transcript cleanup

This would already make the paper much stronger than its current form.

## 8. How This Differs From Prior Work

This evaluation plan should explicitly position the study relative to earlier pipeline papers.

Recommended claim:
- prior note-to-EHR studies usually evaluate extraction quality well but often do not quantitatively evaluate terminology normalization and final FHIR mapping
- prior FHIR normalization studies show that element-level FHIR evaluation is feasible and should be adopted here
- newer end-to-end semantic pipeline studies suggest that completeness, interoperability, baselines, and ablations are necessary in addition to extraction metrics

This lets the paper argue that its main contribution is not just pipeline construction, but a clearer and more defensible audio-to-FHIR evaluation protocol.

## 9. How to Reframe the Current Paper if Metrics Are Not Ready Yet

If the full evaluation cannot be completed before submission, the paper should avoid overclaiming.

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

Recommended near-term revisions:

1. add explicit benchmark framing around PriMock57
2. add one baseline: `gold transcript -> extraction -> FHIR`
3. add a small manual semantic evaluation subset
4. report normalization accuracy separately from extraction accuracy
5. report schema validation results separately from semantic correctness
6. add ablation for Step 4 transcript cleanup

## 10. Priority Order

If only a few additions are possible, do them in this order:

1. add `WER` against PriMock57 reference transcripts
2. add one baseline: `gold transcript -> extraction -> FHIR`
3. add manual precision/recall on a small annotated subset
4. add normalization accuracy on the same subset
5. add speaker-role accuracy
6. add ablation for Step 4 transcript cleanup

## 11. Short Summary

The strongest version of the paper would evaluate:
- benchmark dataset: `PriMock57`
- baselines: `raw ASR`, `gold transcript`, a naive extraction baseline, and ideally a no-normalization baseline
- metrics: `WER`, role accuracy, extraction precision/recall/F1, relation F1, code accuracy, semantic completeness, interoperability score, and FHIR validation pass rate
- ablations: especially whether transcript cleanup, normalization, and relation assembly improve downstream FHIR extraction

Without these additions, the work is best framed as a structurally validated feasibility pipeline, not a fully evaluated end-to-end clinical NLP system.
