import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from pipeline.step01_ingestion import ingest
from pipeline.step01_ingestion import load as load_ingestion
from pipeline.step01_ingestion import save as save_ingestion
from pipeline.step02_diarization import diarize
from pipeline.step02_diarization import load as load_diarization
from pipeline.step02_diarization import save as save_diarization
from pipeline.step03_transcription import transcribe
from pipeline.step03_transcription import load as load_transcription
from pipeline.step03_transcription import save as save_transcription
from pipeline.step04_postprocessing import postprocess
from pipeline.step04_postprocessing import load as load_postprocessing
from pipeline.step04_postprocessing import save as save_postprocessing
from pipeline.step05_fhir_extraction import extract
from pipeline.step05_fhir_extraction import load as load_extraction
from pipeline.step05_fhir_extraction import save as save_extraction
from pipeline.step06_validation import validate
from pipeline.step06_validation import load as load_validation
from pipeline.step06_validation import save as save_validation


def main() -> None:
    audio_dir = Path(__file__).parent.parent / "references" / "primock57" / "audio"
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    doctor_path = audio_dir / "day1_consultation01_doctor.wav"
    patient_path = audio_dir / "day1_consultation01_patient.wav"

    if not doctor_path.exists() or not patient_path.exists():
        print("Missing audio files in", audio_dir)
        return

    doctor, _ = librosa.load(doctor_path, sr=16_000, mono=True)
    patient, _ = librosa.load(patient_path, sr=16_000, mono=True)

    # Pad shorter track to match longer
    max_len = max(len(doctor), len(patient))
    doctor = np.pad(doctor, (0, max_len - len(doctor)))
    patient = np.pad(patient, (0, max_len - len(patient)))

    mixed = doctor + patient
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak

    mixed = mixed[:16_000 * 60]  # trim to 60s for testing

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    sf.write(tmp_path, mixed, 16_000)

    print(f"Mixed tracks: {doctor_path.name} + {patient_path.name}")
    print(f"Mixed duration: {len(mixed) / 16_000:.2f}s")

    step01_out = outputs_dir / "step01_day1_consultation01.wav"
    if step01_out.exists() and step01_out.with_suffix(".json").exists():
        print("\n--- Ingestion (cached) ---")
        ingestion_result = load_ingestion(step01_out)
    else:
        ingestion_result = ingest(tmp_path)
        save_ingestion(ingestion_result, step01_out)
        print("\n--- Ingestion ---")
    print(f"Processed duration: {ingestion_result.duration_s:.2f}s")
    print(f"Speech ratio:       {ingestion_result.speech_ratio:.3f}")
    print(f"Output: {step01_out}")

    step02_out = outputs_dir / "step02_day1_consultation01.json"
    if step02_out.exists():
        print("\n--- Diarization (cached) ---")
        diarization_result = load_diarization(step02_out)
    else:
        diarization_result = diarize(ingestion_result)
        save_diarization(diarization_result, step02_out)
        print("\n--- Diarization ---")
    print(f"Speakers found: {diarization_result.num_speakers}")
    print(f"Total segments: {len(diarization_result.segments)}")
    print(f"\nFirst 10 segments:")
    for seg in diarization_result.segments[:10]:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s]  {seg.speaker}  ({seg.duration:.2f}s)")
    print(f"Output: {step02_out}")

    step03_out = outputs_dir / "step03_day1_consultation01.json"
    if step03_out.exists():
        print("\n--- Transcription (cached) ---")
        transcription_result = load_transcription(step03_out)
    else:
        transcription_result = transcribe(ingestion_result, diarization_result)
        save_transcription(transcription_result, step03_out)
        print("\n--- Transcription ---")
    print(f"Total segments: {len(transcription_result.segments)}")
    for seg in transcription_result.segments:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s] {seg.speaker}: {seg.text}")
    print(f"Output: {step03_out}")

    step04_out = outputs_dir / "step04_day1_consultation01.json"
    if step04_out.exists():
        print("\n--- Post-processing (cached) ---")
        postprocessing_result = load_postprocessing(step04_out)
    else:
        postprocessing_result = postprocess(transcription_result)
        save_postprocessing(postprocessing_result, step04_out)
        print("\n--- Post-processing ---")
    print(f"Total segments: {len(postprocessing_result.segments)}")
    for seg in postprocessing_result.segments:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s] {seg.speaker_role}: {seg.cleaned_text}")
    print(f"Output: {step04_out}")

    step05_out = outputs_dir / "step05_day1_consultation01.json"
    if step05_out.exists():
        print("\n--- FHIR Extraction (cached) ---")
        extraction_result = load_extraction(step05_out)
    else:
        extraction_result = extract(postprocessing_result)
        save_extraction(extraction_result, step05_out)
        print("\n--- FHIR Extraction ---")
    print(f"SOAP summary: {extraction_result.soap_summary}")
    for rtype, count in extraction_result.resource_counts.items():
        print(f"  {rtype}: {count}")
    print(f"Output: {step05_out}")

    step06_out = outputs_dir / "step06_day1_consultation01.json"
    if step06_out.exists():
        print("\n--- FHIR Validation (cached) ---")
        validation_result = load_validation(step06_out)
    else:
        validation_result = validate(extraction_result)
        save_validation(validation_result, step06_out)
        print("\n--- FHIR Validation ---")
    print(f"Valid: {validation_result.valid}")
    errors = [i for i in validation_result.issues if i.severity == "error"]
    warnings = [i for i in validation_result.issues if i.severity == "warning"]
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")
    for issue in errors:
        print(f"  [ERROR] {issue.resource_type}/{issue.resource_id}: {issue.message}")
    total_resources = sum(validation_result.resource_counts.values())
    provenance_count = len([e for e in validation_result.bundle_with_provenance["entry"] if e["resource"]["resourceType"] == "Provenance"])
    print(f"Bundle: {total_resources} clinical resources + {provenance_count} Provenance")
    print(f"Output: {step06_out}")

    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
