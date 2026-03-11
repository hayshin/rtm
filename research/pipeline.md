# Automated Pipeline: Clinical Audio → HL7 FHIR

---

## Pipeline Overview

```
[1]  Audio Ingestion & Pre-processing
          ↓
[2]  Speaker Diarization
          ↓
[3]  ASR Transcription — Whisper (Medical Fine-tune)
          ↓
[4]  LLM Transcript Post-processing + Speaker Role Assignment
          ↓
[5]  LLM Clinical Extraction → FHIR R4 Bundle
          ↓
[6]  FHIR Validation & Provenance Tracking
```

Steps 5–6 collapse the traditional multi-stage NLP pipeline (SOAP segmentation → NER → negation detection → terminology normalization → FHIR mapping) into two passes. This is motivated by the demonstrated capability of instruction-tuned LLMs to perform all of these sub-tasks jointly when given sufficient context, and by the Infherno result showing that agentic LLM-to-FHIR mapping achieves <2.3% semantic hallucination rate with schema-enforcement (Frei et al., 2025) — comparable to dedicated NLP pipelines at a fraction of the engineering complexity.

---

## Phase 1 — Ingestion & Pre-processing

### Step 1 · Audio Ingestion & Pre-processing

**Tools:** FFmpeg, Librosa, WebRTC VAD

Accepts raw audio from microphones, EHR-integrated recorders, or telephony systems. Normalizes format to 16kHz mono WAV, applies noise reduction, and runs Voice Activity Detection (VAD) to strip silence and non-speech segments before transcription begins. A multi-accent adaptive preprocessing layer should be considered, as most published benchmarks are recorded under ideal studio conditions with native English speakers — a significant gap from real clinical environments (Tran et al., 2022).

> VAD alone can reduce Whisper processing time by 30–50%. Noise in raw clinical audio compounds WER at every downstream stage. Whisper large-v3 demonstrates superior noise robustness at SNR < 10 dB over models trained on clean corpora (Radford et al., 2022). Most published WER benchmarks assume quiet, high-quality recordings and likely underestimate real-world error rates.

---

## Phase 2 — Automatic Speech Recognition

### Step 2 · Speaker Diarization

**Tools:** pyannote.audio 4.x, WhisperX (Whisper + Pyannote), AWS Transcribe Medical, NeMo

Segments the audio into speaker turns, separating physician speech from patient speech. Output is a timestamped transcript with generic speaker labels (`SPEAKER_00`, `SPEAKER_01`). Role assignment (PHYSICIAN/PATIENT) is deferred to Step 4, where the LLM can determine roles from transcript content. WhisperX tightly integrates word-level alignment with pyannote diarization, enabling character-level speaker attribution rather than segment-level only (Metcalf et al., 2024). A counterintuitive finding from comparative evaluation is that general-purpose ASR models achieve significantly better diarization accuracy than medical-specialized models despite lower domain coverage (Tran et al., 2022).

> Clinical diarization error rates (WDER) range from 1.8–13.9% (Tran et al., AMIA 2022). The lowest WDER (1.8%) was achieved by a general-purpose model (Amazon Transcribe), not a medical-specialized variant. Speaker attribution is essential — the same phrase carries different clinical meaning depending on who said it.

---

### Step 3 · ASR Transcription — Whisper (Medical Fine-tune)

**Tools:** Whisper large-v3, United-MedASR (Whisper + Faster Whisper + BART-Base), MedicalWhisper

Transcribes each diarized speaker segment to text using a medical-adapted Whisper model. Outputs a timestamped, speaker-labeled transcript with accurate handling of drug names, anatomical terms, Latin abbreviations, and clinical shorthand. The United-MedASR architecture layers a BART-Base semantic correction model over Whisper to specifically address residual medical terminology errors that base Whisper mispronounces or substitutes (Banerjee et al., 2024). Synthetic data generation from ICD-10, MIMS, and FDA databases is used to enrich the fine-tuning corpus without privacy risks.

WER improvement trajectory across the literature:

- ~35% WER — general ASR on clinical audio (Kodish-Wachs et al., 2018)
- 8.8–10.5% WER — commercial ASR systems in primary care (Tran et al., 2022)
- ~2.5% WER — Whisper large-v3, zero-shot (Radford et al., 2022)
- 0.4–1.0% WER — United-MedASR fine-tuned on synthetic medical data (Banerjee et al., 2024)

> Medical fine-tuned variants achieve WER below 2.5%; domain-adapted stacked architectures (Whisper + BART) approach 0.4% WER on benchmark corpora (Banerjee et al., 2024). However, medical concept recall remains unexpectedly low at 0.48–0.49 (F-recall) despite high precision of 0.95–0.96 against LOINC/SNOMED CT concepts (Tran et al., 2022), motivating the post-processing step. Timestamps from this stage are required for FHIR `Provenance` resources in Step 6.

---

### Step 4 · LLM Transcript Post-processing + Speaker Role Assignment

**Tools:** GPT-4o, Llama 3.3-70B, Phi4-14B, Qwen2.5-14B, ClinicalT5, BART-Base (medical fine-tune), MediNotes

Corrects residual ASR errors using contextual language modeling: fixes mis-transcribed drug names, expands abbreviations, resolves homophones (e.g. _ileum_ vs. _ilium_), removes filler words, and restructures conversational speech into coherent clinical utterances. In the same LLM pass, resolves speaker roles — mapping generic `SPEAKER_00`/`SPEAKER_01` labels to `[PHYSICIAN]`/`[PATIENT]` based on transcript content (e.g. clinical questioning patterns, examination language, symptom reporting). Combining role assignment with post-processing avoids a redundant LLM call and is more accurate since the model can reason from the full transcript context. The BART-Base semantic correction layer (United-MedASR) specifically targets medical terminology substitution errors through fine-tuning on synthetic ICD-10 and FDA vocabulary (Banerjee et al., 2024). For privacy-preserving on-premise deployment, 4-bit quantized open-source models are viable: Llama-3.3-70B achieves the highest benchmark score (0.760 DRAGON utility), while Phi4-14B (0.751) and Qwen2.5-14B (0.748) offer a practical parameter-efficiency trade-off on 12GB VRAM hardware (Builtjes et al., 2025). The MediNotes system demonstrates that combining ASR with LLMs and Retrieval-Augmented Generation (RAG) over a medical knowledge base further reduces domain-specific terminology errors (Saadat et al., 2025). Parameter-efficient fine-tuning via QLoRA enables domain adaptation without full model retraining.

> Prompts must be domain-specific with a medical system context. Critically, translating non-English clinical text into English before LLM processing consistently degrades extraction performance — native-language processing must be preserved (Builtjes et al., 2025). Common ASR error patterns (pronoun deletion, agreement token substitution: "yeah", "okay") contribute more to WER than medical term errors, and must be handled in post-processing (Tran et al., 2022).

---

## Phase 3 — Clinical Extraction & FHIR Generation

### Step 5 · LLM Clinical Extraction → FHIR R5 Bundle

**Tools:** GPT-4o, Llama 3.3-70B, Phi4-14B, Qwen2.5-14B, fhir.resources (Python), Infherno, FHIR-GPT, Smolagents

A single LLM pass over the post-processed, role-labeled transcript performs all of: SOAP classification, named entity recognition, negation detection, relation extraction, terminology normalization, and FHIR resource mapping. The LLM receives the full conversation context and outputs a structured extraction schema (Pydantic) that is programmatically converted to FHIR R5 resources (`Condition`, `MedicationStatement`, `Observation`, `Procedure`, `Encounter`).

**SOAP classification** is implicit in resource type selection: subjective patient complaints → `Condition` (unconfirmed), objective exam/lab findings → `Observation`, diagnoses → `Condition` (confirmed), plan items → `MedicationStatement`/`Procedure`. This context-aware mapping avoids the ambiguity that plagues segment-level classification — "blood pressure 140/90" is correctly typed as `Observation` when measured now versus `Condition` risk factor when reported historically, because the LLM reads full context.

**NER and negation** are handled jointly: "no chest pain" or "denies shortness of breath" maps to `verificationStatus=refuted`; "possible pneumonia" maps to `verificationStatus=unconfirmed`. Relation extraction (drug→dose, symptom→location) is resolved at structured output construction time — dosage fields are populated directly from the same extraction pass. This avoids the 23-point semantic completeness drop that ablation studies show when relation extraction is omitted (Semantic NLP Pipelines, Binghamton, 2024).

**Terminology normalization** is performed best-effort by the LLM: SNOMED CT codes for conditions and procedures, RxNorm codes for medications, LOINC codes for observations. While LLM-generated codes are less reliable than dedicated lookup services (QuickUMLS, live SNOMED CT API), they provide a viable approximation for research. Concept normalization has the single highest isolated impact on interoperability score among all pipeline components — a 0.23-point drop when removed (Semantic NLP Pipelines, 2024) — making this the primary limitation of the LLM-only approach versus dedicated terminology services.

**FHIR mapping** uses `fhir.resources` Python objects for schema-constrained construction. The Infherno system demonstrates that programmatic FHIR object construction (rather than free-form JSON generation) reduces hallucination rates to <2.3% (Frei et al., 2025). NLP2FHIR (Mayo Clinic) establishes the baseline mapping rule inventory: 30 NLP-to-FHIR element rules and 62 content normalization rules, achieving F-score 0.69–0.99 across resource types (Hong et al., 2019). LLM attribute-level mapping accuracy: GPT-4o 67–73% (95% CI), Llama 3.2 405b 43–53% on MIMIC-IV FHIR benchmark (Murcia et al., 2024).

The traditional multi-step alternative — separate SOAP (MedSpaCy/T5), NER (ClinicalBERT F1=0.89), negation (NegEx/BioBERT-RE F1=0.81), normalization (QuickUMLS), and mapping (NLP2FHIR) — achieves semantic completeness 91% and interoperability score 0.88 (Semantic NLP Pipelines, 2024), and remains the stronger approach for production systems or component-level ablation research. The LLM-unified approach trades accuracy ceiling for implementation simplicity and is appropriate for rapid research prototyping.

> SOAP is implicit in FHIR resource type selection — the LLM resolves this during extraction, not as a separate prior step. Approximately 30% of clinical entity mentions are negated; missing these generates clinically dangerous false-positive FHIR resources (Semantic NLP Pipelines, 2024). FHIR requires coded `CodeableConcept` values — without ontology grounding resources are not interoperable, making terminology normalization the weakest point of the LLM-unified approach.

---

## Phase 4 — Validation & Output

### Step 6 · FHIR Validation & Provenance Tracking

**Tools:** fhir.resources (Python schema enforcement), HAPI FHIR Validator (optional, Java), SMART Text2FHIR

Validates each resource against the HL7 FHIR R4 schema using `fhir.resources` Pydantic parsing — catching required field violations, type mismatches, and invalid cardinality. Attaches `Provenance` resources linking each clinical resource back to the source audio file and NLP pipeline version, enabling full auditability. NLP metadata extensions from NLP2FHIR (`confidence_score`, `nlp_system`, `offset`) provide a machine-readable audit trail to the exact character position in the source transcript (Hong et al., 2019). SMART Text2FHIR standardizes three provenance extensions: `nlp-source` (algorithm + version), `derivation-reference` (character offset + length), and `nlp-polarity` (negation boolean), enabling downstream consumers to reconstruct NLP decisions without re-running the pipeline (SMART Text2FHIR, Boston Children's Hospital).

For full profile conformance validation (beyond schema), the HAPI FHIR Validator CLI (Java) can be added as an optional post-processing step. Programmatic schema enforcement during assembly reduces validation failures: hallucination rates below 2.3% have been demonstrated when `fhir.resources` object construction is used (Frei et al., 2025).

> Provenance tracking is the primary evaluation mechanism in a research context. FHIR schema validation catches structural errors; NLP metadata extensions enable semantic auditability. The validation pass also serves as the quantitative evaluation point for the research pipeline — error counts and resource coverage metrics are recorded here.
