# Automated Pipeline: Clinical Audio → HL7 FHIR

---

## Pipeline Overview

```
[1]  Audio Ingestion & Pre-processing
          ↓
[2]  De-identification (HIPAA Layer)
          ↓
[3]  Speaker Diarization
          ↓
[3b] LLM-based Diarization Correction
          ↓
[4]  ASR Transcription — Whisper (Medical Fine-tune)
          ↓
[5]  LLM Transcript Post-processing
          ↓
[6]  SOAP Segmentation & Sentence Classification
          ↓
[7]  Named Entity Recognition (NER)
          ↓
[8]  Negation Detection & Relation Extraction
          ↓
[9]  Terminology Normalization
          ↓
[10] FHIR Resource Mapping & Assembly
          ↓
[11] FHIR Validation & Provenance Tracking
          ↓
[12] Human-in-the-Loop Clinician Review
          ↓
[13] EHR Integration via FHIR API
```

---

## Phase 1 — Ingestion & Pre-processing

### Step 1 · Audio Ingestion & Pre-processing

**Tools:** FFmpeg, Librosa, WebRTC VAD

Accepts raw audio from microphones, EHR-integrated recorders, or telephony systems. Normalizes format to 16kHz mono WAV, applies noise reduction, and runs Voice Activity Detection (VAD) to strip silence and non-speech segments before transcription begins. A multi-accent adaptive preprocessing layer should be considered, as most published benchmarks are recorded under ideal studio conditions with native English speakers — a significant gap from real clinical environments (Tran et al., 2022).

> VAD alone can reduce Whisper processing time by 30–50%. Noise in raw clinical audio compounds WER at every downstream stage. Whisper large-v3 demonstrates superior noise robustness at SNR < 10 dB over models trained on clean corpora (Radford et al., 2022). Most published WER benchmarks assume quiet, high-quality recordings and likely underestimate real-world error rates.

---

### Step 2 · De-identification (HIPAA Layer)

**Tools:** Microsoft Presidio, MIST, PhiloASR

Applies a HIPAA §164.514-compliant data handling layer before any content leaves a secure environment — encrypted storage, access logging, and on-premise processing constraints. Audio-level de-identification (before transcription) is preferable to text-level-only de-identification, as PHI present in acoustic features (speaker voice, timing patterns) is not captured post-ASR. PhiloASR operates in the audio domain for this purpose. Text-level de-identification post-ASR removes names, dates, MRNs, and other PHI before any external API calls.

> Required for legal compliance and for using clinical datasets (e.g. MIMIC-IV) in research evaluation.

---

## Phase 2 — Automatic Speech Recognition

### Step 3 · Speaker Diarization

**Tools:** pyannote.audio 3.x, WhisperX (Whisper + Pyannote), AWS Transcribe Medical, NeMo

Segments the audio into speaker turns, separating physician speech from patient speech. Output is a timestamped transcript annotated with speaker labels (`[PHYSICIAN]`, `[PATIENT]`). WhisperX tightly integrates word-level alignment with pyannote diarization, enabling character-level speaker attribution rather than segment-level only (Metcalf et al., 2024). A counterintuitive finding from comparative evaluation is that general-purpose ASR models achieve significantly better diarization accuracy than medical-specialized models despite lower domain coverage (Tran et al., 2022).

> Clinical diarization error rates (WDER) range from 1.8–13.9% (Tran et al., AMIA 2022). The lowest WDER (1.8%) was achieved by a general-purpose model (Amazon Transcribe), not a medical-specialized variant. Speaker attribution is essential — the same phrase carries different clinical meaning depending on who said it.

---

### Step 3b · LLM-based Diarization Correction

**Tools:** Mistral 7b Instruct v0.2, DiarizationLM (ensemble of ASR-specific fine-tuned models)

Applies a post-processing correction pass over the diarized transcript using a language model fine-tuned to recognize and reassign misattributed speaker turns. Raw diarization systems generate systematic errors that depend heavily on the ASR source; a single LLM fine-tuned on one ASR tool fails to generalize to others. The recommended approach is a three-model ensemble — one model fine-tuned per ASR source (AWS, Azure, WhisperX) — combined for robust cross-ASR correction (Willis et al., 2024).

> Fine-tuned ensemble models reduce deltaCP (speaker concatenation penalty) by 0.93–4.46 points and deltaSA (speaker-attributed WER) by 2.5–5.77 points depending on the ASR source (Willis et al., 2024). Zero-shot LLM diarization correction consistently degrades performance relative to no correction — fine-tuning on ASR-specific error patterns is essential. Labeled training data per ASR tool is required.

---

### Step 4 · ASR Transcription — Whisper (Medical Fine-tune)

**Tools:** Whisper large-v3, United-MedASR (Whisper + Faster Whisper + BART-Base), MedicalWhisper

Transcribes each diarized speaker segment to text using a medical-adapted Whisper model. Outputs a timestamped, speaker-labeled transcript with accurate handling of drug names, anatomical terms, Latin abbreviations, and clinical shorthand. The United-MedASR architecture layers a BART-Base semantic correction model over Whisper to specifically address residual medical terminology errors that base Whisper mispronounces or substitutes (Banerjee et al., 2024). Synthetic data generation from ICD-10, MIMS, and FDA databases is used to enrich the fine-tuning corpus without privacy risks.

WER improvement trajectory across the literature:
- ~35% WER — general ASR on clinical audio (Kodish-Wachs et al., 2018)
- 8.8–10.5% WER — commercial ASR systems in primary care (Tran et al., 2022)
- ~2.5% WER — Whisper large-v3, zero-shot (Radford et al., 2022)
- 0.4–1.0% WER — United-MedASR fine-tuned on synthetic medical data (Banerjee et al., 2024)

> Medical fine-tuned variants achieve WER below 2.5%; domain-adapted stacked architectures (Whisper + BART) approach 0.4% WER on benchmark corpora (Banerjee et al., 2024). However, medical concept recall remains unexpectedly low at 0.48–0.49 (F-recall) despite high precision of 0.95–0.96 against LOINC/SNOMED CT concepts (Tran et al., 2022), motivating the post-processing step. Timestamps from this stage are required for FHIR `Provenance` resources in Step 11.

---

### Step 5 · LLM Transcript Post-processing

**Tools:** GPT-4o, Llama 3.3-70B, Phi4-14B, Qwen2.5-14B, ClinicalT5, BART-Base (medical fine-tune), MediNotes

Corrects residual ASR errors using contextual language modeling: fixes mis-transcribed drug names, expands abbreviations, resolves homophones (e.g. _ileum_ vs. _ilium_), removes filler words, and restructures conversational speech into coherent clinical utterances. The BART-Base semantic correction layer (United-MedASR) specifically targets medical terminology substitution errors through fine-tuning on synthetic ICD-10 and FDA vocabulary (Banerjee et al., 2024). For privacy-preserving on-premise deployment, 4-bit quantized open-source models are viable: Llama-3.3-70B achieves the highest benchmark score (0.760 DRAGON utility), while Phi4-14B (0.751) and Qwen2.5-14B (0.748) offer a practical parameter-efficiency trade-off on 12GB VRAM hardware (Builtjes et al., 2025). The MediNotes system demonstrates that combining ASR with LLMs and Retrieval-Augmented Generation (RAG) over a medical knowledge base further reduces domain-specific terminology errors (Saadat et al., 2025). Parameter-efficient fine-tuning via QLoRA enables domain adaptation without full model retraining.

> Prompts must be domain-specific with a medical system context. Critically, translating non-English clinical text into English before LLM processing consistently degrades extraction performance — native-language processing must be preserved (Builtjes et al., 2025). Common ASR error patterns (pronoun deletion, agreement token substitution: "yeah", "okay") contribute more to WER than medical term errors, and must be handled in post-processing (Tran et al., 2022).

---

## Phase 3 — Clinical NLP & Entity Extraction

### Step 6 · SOAP Segmentation & Sentence Classification

**Tools:** MedSpaCy, scispaCy, T5-large (SOAP fine-tune), Cluster2Sent, AutoScribe, LLM zero-shot classification

Classifies transcript segments into the four SOAP categories: **S**ubjective (patient complaints), **O**bjective (vitals, exam findings), **A**ssessment (diagnoses), **P**lan (medications, referrals). This context determines which FHIR resource type each entity maps to downstream. T5-large fine-tuned on K-SOAP/CliniKnote datasets provides a supervised alternative to zero-shot classification with stronger boundary detection (Saadat et al., 2025). For real-time use, Cluster2Sent employs hybrid extractive-abstractive summarization — clustering critical utterances during live consultation and immediately imposing SOAP structure before the encounter ends (Saadat et al., 2025). AutoScribe applies context-driven dialogue parsing with semantic normalization to handle conversational disfluencies specific to clinical interactions. Quality of SOAP output can be measured with DeepScore (domain-specific clinical relevance), BERTScore (semantic similarity), and ROUGE (content preservation).

> "Blood pressure 140/90" is an `Observation` resource when measured objectively, but a `Condition` risk factor when reported historically. SOAP classification resolves this ambiguity. Errors at this stage propagate to all downstream FHIR resource type assignments.

---

### Step 7 · Named Entity Recognition (NER)

**Tools:** ClinicalBERT, BioBERT, scispaCy `en_core_sci_lg`, cTAKES

Extracts typed clinical entities from each SOAP-classified segment: symptoms, diagnoses, medications (name + dose + route + frequency), procedures, anatomical locations, lab values, vital signs, and temporal expressions. Outputs entity spans with type labels and confidence scores. Dedicated sequence-tagging transformer models (ClinicalBERT, BioBERT) significantly outperform rule-based alternatives: ClinicalBERT achieves NER F1 = 0.89 versus F1 = 0.72 for rule-based systems, an improvement of 17 F1 points (Semantic NLP Pipelines, 2024). Generative LLMs (Llama, Phi series) are architecturally mismatched for token-level NER — their continuous text output format cannot reliably produce token-level span labels — and should not replace dedicated NER models at this step (Builtjes et al., 2025). Apache cTAKES with UIMA type system provides a production-grade alternative with integrated SNOMED CT and RxNorm dictionaries (Hong et al., 2019).

> Transformer-based clinical NER achieves F1 0.89 on MIMIC benchmarks (Semantic NLP Pipelines, 2024) vs. 0.85–0.92 on i2b2. NER F1 alone does not determine final interoperability — concept normalization in Step 9 has an equally large downstream impact.

---

### Step 8 · Negation Detection & Relation Extraction

**Tools:** NegEx, MedSpaCy negation, BioBERT-RE, SMART Text2FHIR (`nlp-polarity` extension)

Detects negation ("no chest pain", "denies shortness of breath") and speculation ("possible pneumonia") around each entity. Extracts relations between entities: drug→dosage, symptom→body-location, diagnosis→certainty. Results map directly to FHIR `clinicalStatus` and `verificationStatus` fields. BioBERT-RE achieves relation extraction F1 = 0.81 (Semantic NLP Pipelines, 2024). The SMART Text2FHIR pipeline encodes negation output as an `nlp-polarity` FHIR modifier extension (boolean) attached to each generated resource, enabling downstream systems to interpret polarity without re-running NLP (SMART Text2FHIR, Boston Children's Hospital). Ablation analysis demonstrates that removing relation extraction reduces semantic completeness from 91% to 68% — a 23-point drop (Semantic NLP Pipelines, 2024).

> Approximately 30% of clinical entity mentions are negated. Skipping this step generates clinically dangerous false-positive FHIR resources. Without relation extraction, semantic completeness of the final FHIR bundle degrades by 23 percentage points (Semantic NLP Pipelines, 2024).

---

### Step 9 · Terminology Normalization

**Tools:** UMLS Metathesaurus (QuickUMLS / MetaMap), SNOMED CT (live API / tool-calling), RxNorm, ICD-10-CM, LOINC, cTAKES dictionary lookup

Maps entity surface text to standard ontology codes: diagnoses → SNOMED CT / ICD-10-CM, medications → RxNorm CUI, procedures → CPT / SNOMED CT, lab and vital observations → LOINC. UMLS serves as the cross-terminology hub. Candidate generation from UMLS followed by contextual similarity ranking (rather than simple dictionary lookup) improves disambiguation of ambiguous clinical terms. Inferno (2025) demonstrates that live SNOMED CT tool-calling during entity mapping — rather than static lookup tables — provides higher accuracy by querying the terminology service with full entity context (Frei et al., 2025). Concept normalization is the single largest driver of downstream interoperability: ablation shows a 0.23-point drop in interoperability score when normalization is removed, larger than the impact of removing relation extraction or validation (Semantic NLP Pipelines, 2024).

> FHIR requires coded values (`CodeableConcept`). Without ontology grounding, generated resources are not interoperable. LOINC is mandatory for `Observation` resources. Concept normalization has the highest isolated impact on interoperability score among all NLP pipeline components (Semantic NLP Pipelines, 2024).

---

## Phase 4 — FHIR Resource Generation

### Step 10 · FHIR Resource Mapping & Assembly

**Tools:** NLP2FHIR, FHIR-GPT, Infherno, HAPI FHIR SDK, fhir.resources (Python), Smolagents

Maps normalized entities and relations to FHIR R4 resources: `Condition`, `MedicationStatement`, `Observation`, `Procedure`, `AllergyIntolerance`, `Encounter`. Populates all mandatory fields: `subject` (Patient reference), `status`, `code` (CodeableConcept), `recorder`, and `effectiveDateTime` (from ASR timestamps). A `FHIR Composition` resource wraps all generated resources as a document-level FHIR Bundle, preserving document semantics alongside individual resources (NLP2FHIR, Hong et al., 2019).

**Mapping rule inventory (NLP2FHIR, Mayo Clinic):** 30 NLP-to-FHIR element mapping rules and 62 content normalization rules, achieving F-score 0.69–0.99 across resource types (`MedicationStatement.medicationCodeableConcept`: F = 0.988; `Medication.form`: F = 0.779). Eleven NLP-specific FHIR extensions capture metadata needed for provenance and confidence: `confidence_score`, `negated_modifier`, `certainty_modifier`, `conditional_modifier`, `nlp_system`, `offset` (source character position), `raw_text`, `nlp_date`, `term_temporal`.

**LLM-based approaches:**
- FHIR-GPT (NEJM AI, 2024): GPT-4 class model for direct clinical narrative → `MedicationStatement` mapping
- Infherno (Frei et al., 2025): Gemini-2.5-Pro + Smolagents ReAct framework; enforces FHIR schema compliance through `fhir.resources` Python object construction rather than free-form JSON generation; SNOMED CT integrated via live tool-calling; achieves 86/132 exact field matches with <2.3% semantic hallucination rate on clinical discharge letters
- LLM attribute-level mapping accuracy: GPT-4o 67–73% (95% CI), Llama 3.2 405b 43–53% on structured MIMIC-IV FHIR benchmark (Murcia et al., 2024)
- Full semantic NLP pipeline (entity extraction → normalization → FHIR): NER F1 0.89, RE F1 0.81, semantic completeness 91%, interoperability score 0.88 (Semantic NLP Pipelines, Binghamton, 2024)

> Agentic code-execution approaches (Infherno) achieve lower semantic hallucination rates (<2.3%) than prompt-only LLM approaches because FHIR schema compliance is enforced programmatically rather than instructed via text. JSON generation without schema constraints causes frequent parsing failures (FHIR-GPT). FHIR-GPT achieves >90% exact match on `MedicationStatement`. NLP2FHIR achieves F-score 0.69–0.99 across resource types on Mayo Clinic EHR data (Hong et al., 2019).

---

### Step 11 · FHIR Validation & Provenance Tracking

**Tools:** HAPI FHIR Validator, HL7 Validator CLI, SMART Text2FHIR, fhir.resources (Python schema enforcement)

Validates each resource against official HL7 FHIR R4 profiles using a conformance engine. Attaches `Provenance` resources linking each field back to the source transcript segment, speaker label, and ASR timestamp — enabling full auditability and clinician review. NLP metadata extensions from NLP2FHIR (`confidence_score`, `nlp_system`, `offset`) provide a machine-readable audit trail to the exact character position in the source transcript (Hong et al., 2019). SMART Text2FHIR adds three standardized extensions: `nlp-source` (algorithm + version), `derivation-reference` (character offset + length in source document), and `nlp-polarity` (negation boolean), enabling downstream FHIR consumers to reconstruct NLP decisions without re-running the pipeline (SMART Text2FHIR, Boston Children's Hospital). Programmatic schema enforcement during assembly (Step 10) reduces validation failures at this stage: hallucination rates below 2.3% have been demonstrated when fhir.resources object construction is used (Frei et al., 2025).

> Provenance tracking is the primary evaluation mechanism in a research context and a clinical accountability requirement in deployment. FHIR schema validation catches structural errors; NLP metadata extensions enable semantic auditability.

---

## Phase 5 — Output & Quality Assurance

### Step 12 · Human-in-the-Loop Clinician Review

**Tools:** SMART on FHIR apps, custom review UI

Presents generated FHIR resources to the clinician before committing to the EHR. Flags low-confidence extractions (via `confidence_score` NLP extension) for mandatory review. All edits, rejections, and confirmations are logged with a full audit trail. Systematic review evidence indicates that clinician trust is low when AI-generated outputs are not presented as editable drafts — live editing workflows are a prerequisite for clinical adoption (Saadat et al., 2025). Hallucinations and critical omissions in generated clinical notes are the primary patient safety concerns; cognitive load from reviewing AI-generated content is a secondary risk. Automated pre-screening with DeepScore (clinical domain relevance), BERTScore (semantic similarity), and ROUGE (content completeness) can prioritize resources most needing human attention before the clinician review queue.

> No AI pipeline should autonomously write to a patient EHR. This is both a patient safety requirement and an FDA regulatory expectation for AI-assisted clinical documentation. Editable draft workflows are essential for clinician trust and adoption (Saadat et al., 2025).

---

### Step 13 · EHR Integration via FHIR API

**Tools:** SMART on FHIR (OAuth 2.0), HAPI FHIR Server, Epic / Cerner FHIR R4 APIs

Pushes validated, clinician-approved FHIR resources to the target EHR via RESTful FHIR API (`POST /Condition`, `PUT /MedicationStatement`, etc.). Supports both batch (`Bundle`) and real-time streaming output modes; the `FHIR Bundle` format for batch submission is natively supported by NLP2FHIR and HAPI FHIR, enabling atomic commit of all resources from a single encounter (Hong et al., 2019). All API responses are logged for audit.

> SMART on FHIR provides the OAuth 2.0 authorization layer required by all major EHR vendors (Epic, Cerner, Oracle Health).
