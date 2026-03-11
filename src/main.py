import argparse
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import pipeline.step01_ingestion as step01
import pipeline.step02_diarization as step02
import pipeline.step03_transcription as step03
import pipeline.step04_postprocessing as step04
import pipeline.step05_fhir_extraction as step05
import pipeline.step06_validation as step06


def cached(label, out_path, run_fn, load_fn, save_fn, *, check=None):
    is_cached = check() if check else out_path.exists()
    if is_cached:
        print(f"\n[{label}] cached")
        return load_fn(out_path)
    result = run_fn()
    save_fn(result, out_path)
    print(f"\n[{label}]")
    return result


def mix_tracks(audio_dir: Path, name: str, trim_s: int | None) -> Path:
    """Mix doctor + patient WAV tracks → temp file. Returns tmp path."""
    doctor_path = audio_dir / f"{name}_doctor.wav"
    patient_path = audio_dir / f"{name}_patient.wav"

    if not doctor_path.exists() or not patient_path.exists():
        raise FileNotFoundError(f"Missing audio files for {name} in {audio_dir}")

    doctor, _ = librosa.load(doctor_path, sr=16_000, mono=True)
    patient, _ = librosa.load(patient_path, sr=16_000, mono=True)

    max_len = max(len(doctor), len(patient))
    doctor = np.pad(doctor, (0, max_len - len(doctor)))
    patient = np.pad(patient, (0, max_len - len(patient)))

    mixed = doctor + patient
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak

    if trim_s is not None:
        mixed = mixed[: 16_000 * trim_s]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    sf.write(tmp_path, mixed, 16_000)

    print(f"Mixed tracks: {doctor_path.name} + {patient_path.name}")
    print(f"Mixed duration: {len(mixed) / 16_000:.2f}s")
    return tmp_path


def run_ingestion(outputs_dir: Path, name: str, tmp_path: Path):
    out = outputs_dir / f"step01_{name}.wav"
    result = cached(
        "Ingestion", out,
        lambda: step01.ingest(tmp_path),
        step01.load, step01.save,
        check=lambda: out.exists() and out.with_suffix(".json").exists(),
    )
    print(f"Processed duration: {result.duration_s:.2f}s")
    print(f"Speech ratio:       {result.speech_ratio:.3f}")
    return result


def run_diarization(outputs_dir: Path, name: str, ingestion):
    out = outputs_dir / f"step02_{name}.json"
    result = cached("Diarization", out, lambda: step02.diarize(ingestion), step02.load, step02.save)
    print(f"Speakers found: {result.num_speakers}")
    print(f"Total segments: {len(result.segments)}")
    for seg in result.segments[:10]:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s]  {seg.speaker}  ({seg.duration:.2f}s)")
    return result


def run_transcription(outputs_dir: Path, name: str, ingestion, diarization):
    out = outputs_dir / f"step03_{name}.json"
    result = cached("Transcription", out, lambda: step03.transcribe(ingestion, diarization), step03.load, step03.save)
    print(f"Total segments: {len(result.segments)}")
    for seg in result.segments:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s] {seg.speaker}: {seg.text}")
    return result


def run_postprocessing(outputs_dir: Path, name: str, transcription):
    out = outputs_dir / f"step04_{name}.json"
    result = cached("Post-processing", out, lambda: step04.postprocess(transcription), step04.load, step04.save)
    print(f"Total segments: {len(result.segments)}")
    for seg in result.segments:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s] {seg.speaker_role}: {seg.cleaned_text}")
    return result


def run_fhir_extraction(outputs_dir: Path, name: str, postprocessing):
    out = outputs_dir / f"step05_{name}.json"
    result = cached("FHIR Extraction", out, lambda: step05.extract(postprocessing), step05.load, step05.save)
    print(f"SOAP summary: {result.soap_summary}")
    for rtype, count in result.resource_counts.items():
        print(f"  {rtype}: {count}")
    return result


def run_validation(outputs_dir: Path, name: str, extraction):
    out = outputs_dir / f"step06_{name}.json"
    result = cached("FHIR Validation", out, lambda: step06.validate(extraction), step06.load, step06.save)
    print(f"Valid: {result.valid}")
    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")
    for issue in errors:
        print(f"  [ERROR] {issue.resource_type}/{issue.resource_id}: {issue.message}")
    total = sum(result.resource_counts.values())
    provenance = len([e for e in result.bundle_with_provenance["entry"] if e["resource"]["resourceType"] == "Provenance"])
    print(f"Bundle: {total} clinical resources + {provenance} Provenance")
    return result


def run(name: str, trim_s: int | None = 60) -> None:
    audio_dir = Path(__file__).parent.parent / "references" / "primock57" / "audio"
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    tmp_path = mix_tracks(audio_dir, name, trim_s)
    try:
        ingestion      = run_ingestion(outputs_dir, name, tmp_path)
        diarization    = run_diarization(outputs_dir, name, ingestion)
        transcription  = run_transcription(outputs_dir, name, ingestion, diarization)
        postprocessing = run_postprocessing(outputs_dir, name, transcription)
        extraction     = run_fhir_extraction(outputs_dir, name, postprocessing)
        run_validation(outputs_dir, name, extraction)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical audio → FHIR pipeline")
    parser.add_argument("consultation", nargs="?", default="day1_consultation01")
    parser.add_argument("--full", action="store_true", help="No 60s trim")
    args = parser.parse_args()
    run(args.consultation, trim_s=None if args.full else 60)


if __name__ == "__main__":
    main()
