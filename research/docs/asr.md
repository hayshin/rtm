# ASR Model Research: Step 3

## Selected Model

**`Na0s/Medical-Whisper-Large-v3`** — fine-tuned on PriMock57 directly.

## Evaluated Models

| Model | Params | WER (PriMock57) | CPU RAM | Notes |
|---|---|---|---|---|
| `Na0s/Medical-Whisper-Large-v3` | 1.55B | **0.19** (val) / 0.24 (test) | ~6GB | Fine-tuned on PriMock57 directly |
| `openai/whisper-large-v3` | 1.55B | 0.33 | ~6GB | Best zero-shot baseline |
| `openai/whisper-large-v3-turbo` | 809M | ~0.35 est. | ~3GB | 6x faster, not medical fine-tuned |
| `distil-whisper/distil-large-v3` | 756M | ~10% general | ~1.5GB | Distilled, fast, no medical fine-tune |
| `Esperanto/Medical-Whisper-large-kvc-fp32-onnx` | large-v2 | 0.19 | variable | PriMock57 trained, ONNX format |
| `Crystalcareai/Whisper-Medicalv1` | 756M | unknown | ~1.6GB | distil-large-v3 base, general medical |
| `google/medasr` | 105M | N/A (dictation only) | ~400MB | Physician dictation, not conversations |

## Runtime Options

- **`transformers` pipeline** — simplest, supports any HF model directly
- **`faster-whisper`** — 4x faster via CTranslate2, requires CT2-format models, INT8 quantization (~1.5GB for large-v3-turbo)
- **ONNX** — good CPU inference, Esperanto model available in this format

## Why `Na0s/Medical-Whisper-Large-v3`

- Trained on the exact PriMock57 dataset we are evaluating against — 42% relative WER improvement over base large-v3 (0.19 vs 0.33)
- Clinical consultation speech (UK primary care, same as our dataset)
- Standard Whisper architecture — compatible with HuggingFace `transformers` out of the box
- 16GB RAM is sufficient for FP32 on CPU

## Why not faster-whisper / turbo

- `Na0s/Medical-Whisper-Large-v3` is not yet available in CTranslate2 format — would need manual conversion
- For a research prototype on 60s test clips, raw speed is not the constraint
- Can switch to faster-whisper + CT2 conversion later for full-corpus runs
