"""Step 6: FHIR Validation + Provenance Tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pipeline.step05_fhir_extraction import FHIRExtractionResult

NLP_SYSTEM_URL = "http://example.org/fhir/StructureDefinition/nlp-system"

# Resource types that have required fields we want to check
_IMPORTABLE = {
    "Condition", "MedicationStatement", "Observation",
    "Procedure", "Encounter", "Bundle",
}


@dataclass
class ValidationIssue:
    severity: str      # "error" | "warning"
    resource_type: str
    resource_id: str
    message: str


@dataclass
class ValidationResult:
    issues: list[ValidationIssue]
    valid: bool                  # True = no errors (warnings OK)
    resource_counts: dict[str, int]
    bundle_with_provenance: dict  # bundle + Provenance resources appended
    source_path: Path


def _validate_resource(resource: dict) -> list[ValidationIssue]:
    """Validate a single resource via fhir.resources Pydantic parsing."""
    issues: list[ValidationIssue] = []
    rtype = resource.get("resourceType", "Unknown")
    rid = resource.get("id", "unknown")

    if rtype not in _IMPORTABLE:
        return issues

    try:
        import importlib
        module = importlib.import_module(f"fhir.resources.R4B.{rtype.lower()}")
        cls = getattr(module, rtype)
        cls.model_validate(resource)
    except ImportError:
        issues.append(ValidationIssue(
            severity="warning",
            resource_type=rtype,
            resource_id=rid,
            message="fhir.resources not available; skipped schema validation",
        ))
    except Exception as e:
        # Pydantic ValidationError or similar — report each sub-error if possible
        try:
            for err in e.errors():  # type: ignore[attr-defined]
                loc = " → ".join(str(x) for x in err["loc"])
                issues.append(ValidationIssue(
                    severity="error",
                    resource_type=rtype,
                    resource_id=rid,
                    message=f"{loc}: {err['msg']}",
                ))
        except AttributeError:
            issues.append(ValidationIssue(
                severity="error",
                resource_type=rtype,
                resource_id=rid,
                message=str(e),
            ))

    return issues


def _build_provenance(resource: dict, source_path: Path, model_id: str) -> dict:
    pipeline_version = f"rtm-pipeline/step05/{model_id}"
    return {
        "resourceType": "Provenance",
        "id": str(uuid4()),
        "target": [{"reference": f"{resource['resourceType']}/{resource['id']}"}],
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": [
            {
                "type": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
                        "code": "assembler",
                    }]
                },
                "who": {"display": pipeline_version},
            }
        ],
        "entity": [
            {
                "role": "source",
                "what": {"display": source_path.name},
            }
        ],
        "extension": [
            {
                "url": NLP_SYSTEM_URL,
                "valueString": pipeline_version,
            }
        ],
    }


def validate(extraction: FHIRExtractionResult) -> ValidationResult:
    all_issues: list[ValidationIssue] = _validate_resource(extraction.bundle)
    provenance_resources: list[dict] = []

    for entry in extraction.bundle.get("entry", []):
        resource = entry.get("resource", {})
        if not resource or not resource.get("id"):
            continue

        issues = _validate_resource(resource)
        all_issues.extend(issues)
        provenance_resources.append(_build_provenance(resource, extraction.source_path, extraction.model_id))

    bundle_with_provenance = dict(extraction.bundle)
    bundle_with_provenance["entry"] = list(extraction.bundle.get("entry", [])) + [
        {"resource": p} for p in provenance_resources
    ]

    return ValidationResult(
        issues=all_issues,
        valid=not any(i.severity == "error" for i in all_issues),
        resource_counts=extraction.resource_counts,
        bundle_with_provenance=bundle_with_provenance,
        source_path=extraction.source_path,
    )


def save(result: ValidationResult, out_path: Path) -> None:
    data = {
        "source_path": str(result.source_path),
        "valid": result.valid,
        "resource_counts": result.resource_counts,
        "issues": [asdict(i) for i in result.issues],
        "bundle_with_provenance": result.bundle_with_provenance,
    }
    out_path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> ValidationResult:
    data = json.loads(path.read_text())
    return ValidationResult(
        issues=[ValidationIssue(**i) for i in data["issues"]],
        valid=data["valid"],
        resource_counts=data["resource_counts"],
        bundle_with_provenance=data["bundle_with_provenance"],
        source_path=Path(data["source_path"]),
    )
