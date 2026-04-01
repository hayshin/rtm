# RTM

This project can now use an NVIDIA RTX GPU for diarization and transcription.

## RTX setup

1. Install the NVIDIA driver on the host and confirm the GPU is visible:

```bash
nvidia-smi
```

2. Install a CUDA-enabled PyTorch build in this project environment.
Use the official PyTorch install selector for your OS and CUDA version, then sync the environment again if needed.

3. Install project dependencies:

```bash
uv sync
```

4. Set the Hugging Face token required by pyannote:

```bash
export HF_TOKEN=your_token_here
```

5. Select the runtime device. The default is `auto`, which uses CUDA when `torch.cuda.is_available()` is true.

```bash
export RTM_DEVICE=auto
```

Optional:

```bash
export RTM_DEVICE=cuda
export RTM_COMPUTE_TYPE=float16
```

`RTM_DEVICE` values:

- `auto`: prefer CUDA, otherwise CPU
- `cuda`: require CUDA and fail if unavailable
- `cpu`: force CPU execution

`RTM_COMPUTE_TYPE` controls the `faster-whisper` compute mode. Typical values are `float16` for RTX GPUs and `int8` for CPU.

## Transcription backend

Step 3 now defaults to the Hugging Face transformers backend with
`Na0s/Medical-Whisper-Large-v3` so the pipeline matches the Medical Whisper
setup described in the paper.

Before loading the ASR model, the pipeline explicitly clears reserved CUDA
memory to reduce out-of-memory failures after GPU diarization.

## Verify CUDA is active

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If this prints `True` and your GPU name, the project will use the RTX card when `RTM_DEVICE=auto` or `RTM_DEVICE=cuda`.
