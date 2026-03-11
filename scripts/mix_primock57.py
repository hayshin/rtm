"""Mix doctor + patient audio tracks from the PriMock57 dataset.

For each consultation, loads the separate doctor and patient WAV files,
sums them together, normalises, and writes a mixed WAV to the output dir.

Usage:
    uv run python scripts/mix_primock57.py                         # all consultations
    uv run python scripts/mix_primock57.py day1_consultation01     # one consultation
    uv run python scripts/mix_primock57.py --trim 120              # trim to 2 min
    uv run python scripts/mix_primock57.py -o /tmp/mixed           # custom output dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_DIR = Path(__file__).parent.parent / "references" / "primock57" / "audio"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "mixed"
SR = 16_000


def mix_pair(name: str, audio_dir: Path, out_dir: Path, trim_s: int | None) -> Path:
    doctor_path = audio_dir / f"{name}_doctor.wav"
    patient_path = audio_dir / f"{name}_patient.wav"

    if not doctor_path.exists():
        raise FileNotFoundError(f"Missing: {doctor_path}")
    if not patient_path.exists():
        raise FileNotFoundError(f"Missing: {patient_path}")

    doctor, _ = librosa.load(doctor_path, sr=SR, mono=True)
    patient, _ = librosa.load(patient_path, sr=SR, mono=True)

    max_len = max(len(doctor), len(patient))
    doctor = np.pad(doctor, (0, max_len - len(doctor)))
    patient = np.pad(patient, (0, max_len - len(patient)))

    mixed = doctor + patient
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak

    if trim_s is not None:
        mixed = mixed[: SR * trim_s]

    out_path = out_dir / f"{name}_mixed.wav"
    sf.write(out_path, mixed, SR)
    return out_path


def find_consultations(audio_dir: Path) -> list[str]:
    names = sorted(
        p.stem.removesuffix("_doctor")
        for p in audio_dir.glob("*_doctor.wav")
    )
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix PriMock57 doctor+patient tracks")
    parser.add_argument(
        "consultation",
        nargs="?",
        default=None,
        help="Consultation name, e.g. day1_consultation01 (default: all)",
    )
    parser.add_argument(
        "--trim",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Trim output to this many seconds (default: no trim)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help=f"Input audio directory (default: {AUDIO_DIR})",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.consultation:
        names = [args.consultation]
    else:
        names = find_consultations(args.audio_dir)
        if not names:
            print(f"No *_doctor.wav files found in {args.audio_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(names)} consultations")

    errors: list[tuple[str, str]] = []
    for name in names:
        try:
            out_path = mix_pair(name, args.audio_dir, args.output, args.trim)
            duration = librosa.get_duration(path=out_path)
            print(f"  {name} → {out_path.name}  ({duration:.1f}s)")
        except Exception as exc:
            print(f"  {name} → ERROR: {exc}", file=sys.stderr)
            errors.append((name, str(exc)))

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for name, msg in errors:
            print(f"  {name}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
