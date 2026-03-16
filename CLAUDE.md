# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

NixOS + devenv + direnv. The devenv shell auto-activates via `.envrc`. On NixOS, C extension libraries require `LD_LIBRARY_PATH` (set in `devenv.nix`).

Running any Python command requires the devenv environment:
```bash
eval "$(direnv export bash)" && uv run python src/main.py
```

## Commands

```bash
# Run CLI pipeline (default: day1_consultation01, trimmed to 60s)
eval "$(direnv export bash)" && uv run python src/main.py [consultation_name] [--full]

# Launch Gradio web UI
eval "$(direnv export bash)" && uv run python src/app.py

# Add a dependency
uv add <package>
```

## Required Environment Variables

- `HF_TOKEN` — HuggingFace token for pyannote diarization model (step 2)
- `OPENAI_API_KEY` — OpenAI key for LLM steps (steps 4 and 5)

These are loaded from `.env` via `dotenv.enable = true` in `devenv.nix`.

## Architecture

**Clinical audio → FHIR R4 bundle pipeline** operating on PriMock57 dataset (57 mock GP consultations, doctor + patient as separate WAV tracks).

Input audio has two tracks per consultation (`{name}_doctor.wav` + `{name}_patient.wav`) in `references/primock57/audio/`. `main.py:mix_tracks()` merges these into a single temp WAV before running the pipeline.

### Pipeline Steps (`src/pipeline/`)

| Step | File | Description | Key dependency |
|------|------|-------------|----------------|
| 1 | `step01_ingestion.py` | Load audio → 16kHz mono → noise reduce → WebRTC VAD | librosa, noisereduce, webrtcvad |
| 2 | `step02_diarization.py` | Speaker diarization | pyannote/speaker-diarization-community-1 (HF_TOKEN required) |
| 3 | `step03_transcription.py` | ASR transcription per diarized segment | Na0s/Medical-Whisper-Large-v3 (HuggingFace transformers) |
| 4 | `step04_postprocessing.py` | LLM cleans transcript + assigns PHYSICIAN/PATIENT roles | agno + OpenAI |
| 5 | `step05_fhir_extraction.py` | LLM extracts FHIR R4 resources (Condition, Medication, Observation, Procedure) | agno + OpenAI |
| 6 | `step06_validation.py` | Validates resources via fhir.resources Pydantic, appends Provenance | fhir-resources |

### Module Contract

Every step module exposes the same interface:
- `run_fn(inputs) → Result` — compute the step
- `save(result, path)` — serialize to disk
- `load(path) → Result` — deserialize from disk

`Result` types are `@dataclass` instances; LLM-structured outputs use Pydantic `BaseModel`.

### Caching

`main.py` wraps each step in `cached()`: if the output file exists, `load()` is called instead of re-running. CLI outputs go to `outputs/step0N_{name}.*`; the Gradio UI (`src/app.py`) caches per-consultation under `outputs/{name}/step0N.*`.

### Two Entry Points

- **`src/main.py`** — CLI, uses hardcoded `references/primock57/audio/` as audio source, handles the doctor+patient track mixing
- **`src/app.py`** — Gradio web UI, accepts any uploaded audio file, streams step-by-step status updates

### FHIR Resource Mapping

Step 5 extracts into an R4 Bundle (type: `collection`) with: `Encounter`, `Condition` (SNOMED/ICD-10), `MedicationStatement` (RxNorm), `Observation` (LOINC), `Procedure` (SNOMED). Each resource carries a custom extension `source-segment-indices` linking back to transcript segments. Step 6 adds a `Provenance` resource per clinical resource.
