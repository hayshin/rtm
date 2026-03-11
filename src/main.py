from pathlib import Path

import librosa

from pipeline.step01_ingestion import ingest


def main() -> None:
    audio_dir = Path(__file__).parent.parent / "references" / "primock57" / "audio"
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found in", audio_dir)
        return

    path = wav_files[0]
    original_duration = librosa.get_duration(path=path)

    print(f"Source:             {path.name}")
    print(f"Original duration:  {original_duration:.2f}s")

    result = ingest(path)

    print(f"Processed duration: {result.duration_s:.2f}s")
    print(f"Speech ratio:       {result.speech_ratio:.3f}")
    print(f"Output shape:       {result.samples.shape}")


if __name__ == "__main__":
    main()
