"""Step 5: LLM Clinical Extraction → FHIR R4 Bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from pipeline.step04_postprocessing import PostProcessingResult

PATIENT_ID = "patient-001"
PRACTITIONER_ID = "practitioner-001"
ENCOUNTER_DATE = "2024-01-01"  # placeholder for research

SYSTEM_PROMPT = """You are a clinical NLP specialist. Given a post-processed doctor-patient consultation transcript with PHYSICIAN/PATIENT speaker labels, extract all clinically relevant information for FHIR R4 resource creation.

For each entity:
- SOAP is implicit in resource type: symptoms/diagnoses → Condition, vitals/labs → Observation, medications → MedicationStatement, procedures → Procedure
- Detect negation: "no chest pain", "denies X" → verification_status="refuted"; "possible" → "unconfirmed"
- Provide best-effort terminology codes: SNOMED CT for conditions/procedures, RxNorm for medications, LOINC for observations. Use null if uncertain.
- Record segment_indices (list of segment numbers) that support each entity.

Rules:
- Only extract entities explicitly mentioned. No inference.
- Same entity mentioned multiple times = one entry, merge segment_indices.
- Past medications → status "stopped"; current → "active"."""


class _Condition(BaseModel):
    text: str
    clinical_status: str = "unknown"       # "active" | "resolved" | "unknown"
    verification_status: str = "unconfirmed"  # "confirmed" | "unconfirmed" | "refuted"
    snomed_code: str | None = None
    snomed_display: str | None = None
    icd10_code: str | None = None
    segment_indices: list[int] = []


class _Medication(BaseModel):
    drug_name: str
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    status: str = "unknown"  # "active" | "stopped" | "unknown"
    rxnorm_code: str | None = None
    segment_indices: list[int] = []


class _Observation(BaseModel):
    text: str
    value: str | None = None
    unit: str | None = None
    loinc_code: str | None = None
    loinc_display: str | None = None
    segment_indices: list[int] = []


class _Procedure(BaseModel):
    text: str
    status: str = "completed"  # "completed" | "not-done" | "planned"
    snomed_code: str | None = None
    snomed_display: str | None = None
    segment_indices: list[int] = []


class _LLMExtractionResult(BaseModel):
    conditions: list[_Condition] = []
    medications: list[_Medication] = []
    observations: list[_Observation] = []
    procedures: list[_Procedure] = []
    soap_summary: str = ""  # 2–4 sentence SOAP note


@dataclass
class FHIRExtractionResult:
    bundle: dict          # FHIR R4 Bundle as plain dict
    resource_counts: dict[str, int]
    soap_summary: str
    source_path: Path


# ── FHIR R4 resource builders ─────────────────────────────────────────────────

def _codesystem_clinical() -> str:
    return "http://terminology.hl7.org/CodeSystem/condition-clinical"

def _codesystem_ver() -> str:
    return "http://terminology.hl7.org/CodeSystem/condition-ver-status"


def _build_condition(c: _Condition, encounter_ref: str) -> dict:
    coding = []
    if c.snomed_code:
        coding.append({
            "system": "http://snomed.info/sct",
            "code": c.snomed_code,
            "display": c.snomed_display or c.text,
        })
    if c.icd10_code:
        coding.append({"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": c.icd10_code})
    if not coding:
        coding.append({"display": c.text})

    cs = c.clinical_status if c.clinical_status in ("active", "resolved", "inactive", "remission", "unknown") else "unknown"
    vs = c.verification_status if c.verification_status in ("confirmed", "unconfirmed", "refuted", "provisional", "differential", "entered-in-error") else "unconfirmed"

    return {
        "resourceType": "Condition",
        "id": str(uuid4()),
        "clinicalStatus": {
            "coding": [{"system": _codesystem_clinical(), "code": cs}]
        },
        "verificationStatus": {
            "coding": [{"system": _codesystem_ver(), "code": vs}]
        },
        "code": {"coding": coding, "text": c.text},
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "encounter": {"reference": encounter_ref},
    }


def _build_medication(m: _Medication, encounter_ref: str) -> dict:
    coding = []
    if m.rxnorm_code:
        coding.append({
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": m.rxnorm_code,
            "display": m.drug_name,
        })
    else:
        coding.append({"display": m.drug_name})

    valid_statuses = ("active", "completed", "entered-in-error", "intended", "stopped", "on-hold", "unknown", "not-taken")
    status = m.status if m.status in valid_statuses else "unknown"

    resource: dict = {
        "resourceType": "MedicationStatement",
        "id": str(uuid4()),
        "status": status,
        "medicationCodeableConcept": {"coding": coding, "text": m.drug_name},
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "context": {"reference": encounter_ref},
    }

    dosage: dict = {}
    if m.dose:
        dosage["text"] = m.dose
    if m.route:
        dosage["route"] = {"text": m.route}
    if m.frequency:
        dosage["timing"] = {"code": {"text": m.frequency}}
    if dosage:
        resource["dosage"] = [dosage]

    return resource


def _build_observation(o: _Observation, encounter_ref: str) -> dict:
    coding = []
    if o.loinc_code:
        coding.append({
            "system": "http://loinc.org",
            "code": o.loinc_code,
            "display": o.loinc_display or o.text,
        })
    else:
        coding.append({"display": o.text})

    resource: dict = {
        "resourceType": "Observation",
        "id": str(uuid4()),
        "status": "final",
        "code": {"coding": coding, "text": o.text},
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "encounter": {"reference": encounter_ref},
    }
    if o.value:
        resource["valueString"] = f"{o.value} {o.unit or ''}".strip()

    return resource


def _build_procedure(p: _Procedure, encounter_ref: str) -> dict:
    coding = []
    if p.snomed_code:
        coding.append({
            "system": "http://snomed.info/sct",
            "code": p.snomed_code,
            "display": p.snomed_display or p.text,
        })
    else:
        coding.append({"display": p.text})

    valid_statuses = ("preparation", "in-progress", "not-done", "on-hold", "stopped", "completed", "entered-in-error", "unknown")
    status = p.status if p.status in valid_statuses else "completed"

    return {
        "resourceType": "Procedure",
        "id": str(uuid4()),
        "status": status,
        "code": {"coding": coding, "text": p.text},
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "encounter": {"reference": encounter_ref},
    }


# ── Main function ─────────────────────────────────────────────────────────────

def extract(
    postprocessing: PostProcessingResult,
    *,
    model_id: str = "gpt-4o-mini",
    openai_api_key: str | None = None,
) -> FHIRExtractionResult:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    agent = Agent(
        model=OpenAIChat(id=model_id, api_key=api_key),
        instructions=SYSTEM_PROMPT,
        response_model=_LLMExtractionResult,
    )

    lines = ["Full transcript (post-processed):"]
    for i, seg in enumerate(postprocessing.segments):
        lines.append(f"[{i}] {seg.speaker_role} ({seg.start:.2f}s–{seg.end:.2f}s): {seg.cleaned_text}")

    response = agent.run("\n".join(lines))
    extracted: _LLMExtractionResult = response.content

    # Encounter wraps all resources
    encounter_id = str(uuid4())
    encounter_ref = f"Encounter/{encounter_id}"

    segs = postprocessing.segments
    period_start = f"{ENCOUNTER_DATE}T{int(segs[0].start // 3600):02d}:{int((segs[0].start % 3600) // 60):02d}:{int(segs[0].start % 60):02d}Z" if segs else f"{ENCOUNTER_DATE}T00:00:00Z"
    period_end = f"{ENCOUNTER_DATE}T{int(segs[-1].end // 3600):02d}:{int((segs[-1].end % 3600) // 60):02d}:{int(segs[-1].end % 60):02d}Z" if segs else f"{ENCOUNTER_DATE}T00:00:00Z"

    encounter = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "AMB",
            "display": "ambulatory",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "participant": [{"individual": {"reference": f"Practitioner/{PRACTITIONER_ID}"}}],
        "period": {"start": period_start, "end": period_end},
    }

    resources = [encounter]
    for c in extracted.conditions:
        resources.append(_build_condition(c, encounter_ref))
    for m in extracted.medications:
        resources.append(_build_medication(m, encounter_ref))
    for o in extracted.observations:
        resources.append(_build_observation(o, encounter_ref))
    for p in extracted.procedures:
        resources.append(_build_procedure(p, encounter_ref))

    bundle: dict = {
        "resourceType": "Bundle",
        "id": str(uuid4()),
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }

    counts = {
        "Encounter": 1,
        "Condition": len(extracted.conditions),
        "MedicationStatement": len(extracted.medications),
        "Observation": len(extracted.observations),
        "Procedure": len(extracted.procedures),
    }

    return FHIRExtractionResult(
        bundle=bundle,
        resource_counts=counts,
        soap_summary=extracted.soap_summary,
        source_path=postprocessing.source_path,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save(result: FHIRExtractionResult, out_path: Path) -> None:
    data = {
        "source_path": str(result.source_path),
        "resource_counts": result.resource_counts,
        "soap_summary": result.soap_summary,
        "bundle": result.bundle,
    }
    out_path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> FHIRExtractionResult:
    data = json.loads(path.read_text())
    return FHIRExtractionResult(
        bundle=data["bundle"],
        resource_counts=data["resource_counts"],
        soap_summary=data["soap_summary"],
        source_path=Path(data["source_path"]),
    )
