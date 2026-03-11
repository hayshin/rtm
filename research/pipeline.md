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

Accepts raw audio from microphones, EHR-integrated recorders, or telephony systems. Normalizes format to 16kHz mono WAV, applies noise reduction, and runs Voice Activity Detection (VAD) to strip silence and non-speech segments before transcription begins.

> VAD alone can reduce Whisper processing time by 30–50%. Noise in raw clinical audio compounds WER at every downstream stage.

---

### Step 2 · De-identification (HIPAA Layer)

**Tools:** Microsoft Presidio, MIST, PhiloASR

Applies a HIPAA §164.514-compliant data handling layer before any content leaves a secure environment — encrypted storage, access logging, and on-premise processing constraints. Text-level de-identification post-ASR removes names, dates, MRNs, and other PHI before any external API calls.

> Required for legal compliance and for using clinical datasets (e.g. MIMIC-IV) in research evaluation.

---

## Phase 2 — Automatic Speech Recognition

### Step 3 · Speaker Diarization

**Tools:** pyannote.audio 3.x, AWS Transcribe Medical, NeMo

Segments the audio into speaker turns, separating physician speech from patient speech. Output is a timestamped transcript annotated with speaker labels (`[PHYSICIAN]`, `[PATIENT]`).

> Clinical diarization error rates range from 1.8–13.9% (AMIA, 2022). Speaker attribution is essential — the same phrase carries different clinical meaning depending on who said it.

---

### Step 4 · ASR Transcription — Whisper (Medical Fine-tune)

**Tools:** Whisper large-v3, MedicalWhisper, United-MedASR

Transcribes each diarized speaker segment to text using a medical-adapted Whisper model. Outputs a timestamped, speaker-labeled transcript with accurate handling of drug names, anatomical terms, Latin abbreviations, and clinical shorthand.

> Medical fine-tuned variants achieve WER below 5%, compared to ~12% for generic Whisper on clinical audio. Timestamps are required for FHIR provenance in Step 11.

---

### Step 5 · LLM Transcript Post-processing

**Tools:** GPT-4o, Llama 3.3-70B, ClinicalT5

Corrects residual ASR errors using contextual language modeling: fixes mis-transcribed drug names, expands abbreviations, resolves homophones (e.g. _ileum_ vs. _ilium_), removes filler words, and restructures conversational speech into coherent clinical utterances. Prompts must be domain-specific with a medical system context.

---

## Phase 3 — Clinical NLP & Entity Extraction

### Step 6 · SOAP Segmentation & Sentence Classification

**Tools:** MedSpaCy, scispaCy, LLM zero-shot classification

Classifies transcript segments into the four SOAP categories: **S**ubjective (patient complaints), **O**bjective (vitals, exam findings), **A**ssessment (diagnoses), **P**lan (medications, referrals). This context determines which FHIR resource type each entity maps to downstream.

> "Blood pressure 140/90" is an `Observation` resource when measured objectively, but a `Condition` risk factor when reported historically. SOAP classification resolves this ambiguity.

---

### Step 7 · Named Entity Recognition (NER)

**Tools:** ClinicalBERT, BioBERT, scispaCy `en_core_sci_lg`

Extracts typed clinical entities from each SOAP-classified segment: symptoms, diagnoses, medications (name + dose + route + frequency), procedures, anatomical locations, lab values, vital signs, and temporal expressions. Outputs entity spans with type labels and confidence scores.

> Transformer-based clinical NER achieves F1 0.85–0.92 on i2b2 and MIMIC benchmarks.

---

### Step 8 · Negation Detection & Relation Extraction

**Tools:** NegEx, MedSpaCy negation, BioBERT-RE

Detects negation ("no chest pain", "denies shortness of breath") and speculation ("possible pneumonia") around each entity. Extracts relations between entities: drug→dosage, symptom→body-location, diagnosis→certainty. Results map directly to FHIR `clinicalStatus` and `verificationStatus` fields.

> Approximately 30% of clinical entity mentions are negated. Skipping this step generates clinically dangerous false-positive FHIR resources.

---

### Step 9 · Terminology Normalization

**Tools:** UMLS Metathesaurus, SNOMED CT, RxNorm, ICD-10-CM, LOINC

Maps entity surface text to standard ontology codes: diagnoses → SNOMED CT / ICD-10-CM, medications → RxNorm CUI, procedures → CPT / SNOMED CT, lab and vital observations → LOINC. UMLS serves as the cross-terminology hub.

> FHIR requires coded values (`CodeableConcept`). Without ontology grounding, generated resources are not interoperable. LOINC is mandatory for `Observation` resources.

---

## Phase 4 — FHIR Resource Generation

### Step 10 · FHIR Resource Mapping & Assembly

**Tools:** NLP2FHIR, FHIR-GPT, Infherno, HAPI FHIR SDK

Maps normalized entities and relations to FHIR R4 resources: `Condition`, `MedicationStatement`, `Observation`, `Procedure`, `AllergyIntolerance`, `Encounter`. Populates all mandatory fields: `subject` (Patient reference), `status`, `code` (CodeableConcept), `recorder`, and `effectiveDateTime` (from ASR timestamps).

> FHIR-GPT achieves >90% exact match on `MedicationStatement`. NLP2FHIR achieves F-score 0.69–0.99 across resource types on Mayo Clinic EHR data.

---

### Step 11 · FHIR Validation & Provenance Tracking

**Tools:** HAPI FHIR Validator, HL7 Validator CLI, SMART Text2FHIR

Validates each resource against official HL7 FHIR R4 profiles using a conformance engine. Attaches `Provenance` resources linking each field back to the source transcript segment, speaker label, and ASR timestamp — enabling full auditability and clinician review.

> Provenance tracking is the primary evaluation mechanism in a research context and a clinical accountability requirement in deployment.

---

## Phase 5 — Output & Quality Assurance

### Step 12 · Human-in-the-Loop Clinician Review

**Tools:** SMART on FHIR apps, custom review UI

Presents generated FHIR resources to the clinician before committing to the EHR. Flags low-confidence extractions for mandatory review. All edits, rejections, and confirmations are logged with a full audit trail.

> No AI pipeline should autonomously write to a patient EHR. This is both a patient safety requirement and an FDA regulatory expectation for AI-assisted clinical documentation.

---

### Step 13 · EHR Integration via FHIR API

**Tools:** SMART on FHIR (OAuth 2.0), HAPI FHIR Server, Epic / Cerner FHIR R4 APIs

Pushes validated, clinician-approved FHIR resources to the target EHR via RESTful FHIR API (`POST /Condition`, `PUT /MedicationStatement`, etc.). Supports both batch (`Bundle`) and real-time streaming output modes. All API responses are logged for audit.

> SMART on FHIR provides the OAuth 2.0 authorization layer required by all major EHR vendors (Epic, Cerner, Oracle Health).
