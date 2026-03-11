import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from pipeline.step01_ingestion import ingest
from pipeline.step01_ingestion import save as save_ingestion
from pipeline.step02_diarization import diarize
from pipeline.step02_diarization import save as save_diarization


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

    ingestion_result = ingest(tmp_path)
    print(f"\n--- Ingestion ---")
    print(f"Processed duration: {ingestion_result.duration_s:.2f}s")
    print(f"Speech ratio:       {ingestion_result.speech_ratio:.3f}")
    step01_out = outputs_dir / "step01_day1_consultation01.wav"
    save_ingestion(ingestion_result, step01_out)
    print(f"Saved: {step01_out}")

    diarization_result = diarize(ingestion_result)
    print(f"\n--- Diarization ---")
    print(f"Speakers found: {diarization_result.num_speakers}")
    print(f"Total segments: {len(diarization_result.segments)}")
    print(f"\nFirst 10 segments:")
    for seg in diarization_result.segments[:10]:
        print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s]  {seg.speaker}  ({seg.duration:.2f}s)")
    step02_out = outputs_dir / "step02_day1_consultation01.json"
    save_diarization(diarization_result, step02_out)
    print(f"Saved: {step02_out}")

    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
