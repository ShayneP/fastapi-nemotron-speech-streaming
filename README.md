# OpenAI-Compatible STT Server (Nemotron Speech Streaming)

A FastAPI server that wraps NVIDIA's [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) model behind an OpenAI-compatible `/v1/audio/transcriptions` API.

The model uses NeMo's cache-aware FastConformer-RNNT architecture (600M params) and supports true streaming transcription via Server-Sent Events.

## Setup

```bash
# Install dependencies (handles NeMo + triton workaround on Mac)
./install.sh

# Activate the venv
source .venv/bin/activate
```

## Usage

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python server.py
```

The server starts on `http://localhost:8000`. Pass `--host` and `--port` to customize.

## API

### `POST /v1/audio/transcriptions`

OpenAI-compatible transcription endpoint. Accepts multipart form data.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Audio file (wav, mp3, m4a, etc.) |
| `model` | string | yes | Model name (any value accepted) |
| `response_format` | string | no | `"json"` (default), `"text"`, or `"verbose_json"` |
| `stream` | string | no | `"true"` to enable SSE streaming |
| `language` | string | no | Ignored (English only) |
| `temperature` | string | no | Ignored |
| `prompt` | string | no | Ignored |

**Non-streaming:**

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=nemotron-speech-streaming
```

```json
{"text": "The transcribed text."}
```

**Streaming (SSE):**

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=nemotron-speech-streaming \
  -F stream=true
```

```
data: {"type": "transcript.text.delta", "delta": "The "}
data: {"type": "transcript.text.delta", "delta": "transcribed "}
data: {"type": "transcript.text.delta", "delta": "text."}
data: {"type": "transcript.text.done", "text": "The transcribed text."}
data: [DONE]
```

### `GET /v1/models`

Lists available models (for client compatibility).

### `GET /health`

Returns server and model status.

## LiveKit Agents

Works as a drop-in STT provider for LiveKit Agents using the [OpenAI plugin](https://docs.livekit.io/agents/models/stt/plugins/openai/):

```python
from livekit.agents import AgentSession
from livekit.plugins import openai

session = AgentSession(
    stt=openai.STT(
        model="nemotron-speech-streaming",
        base_url="http://localhost:8000/v1",
        api_key="unused",
    ),
    # ... llm, tts, etc.
)
```

## How it works

The server bypasses NeMo's high-level `transcribe()` API (which has a lhotse dataloader dependency that breaks on some setups) and instead runs inference directly:

1. **Audio preprocessing** — Uploaded audio is decoded, converted to 16kHz mono float32
2. **Mel spectrogram** — `model.preprocessor` converts raw audio to mel features
3. **Encoding** — `model.encoder` runs the FastConformer encoder
4. **Decoding** — `model.decoding.rnnt_decoder_predictions_tensor` runs the RNNT decoder to produce text

For streaming, the server uses `model.conformer_stream_step()` which processes audio in chunks using the model's cache-aware streaming config, carrying forward encoder cache state and RNNT hypotheses between steps to produce incremental transcript deltas.
