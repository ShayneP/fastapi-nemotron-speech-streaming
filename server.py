"""
OpenAI-compatible STT server wrapping NVIDIA's nemotron-speech-streaming-en-0.6b model.

Usage:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python server.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger("stt-server")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
MODEL_ID = "nemotron-speech-streaming"
TARGET_SAMPLE_RATE = 16000

asr_model = None


def load_model():
    global asr_model
    logger.info("Loading model %s ...", MODEL_NAME)
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    asr_model.eval()

    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        logger.info("Model on CUDA")
    elif torch.backends.mps.is_available():
        try:
            asr_model = asr_model.to("mps")
            logger.info("Model on MPS")
        except Exception:
            logger.info("MPS failed, using CPU")
    else:
        logger.info("Model on CPU")

    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="NeMo STT Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def load_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Load audio bytes, resample to 16kHz mono, return float32 numpy array."""
    suffix = os.path.splitext(filename)[1] if filename else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    try:
        data, sr = sf.read(tmp_in_path, dtype="float32")
    except Exception:
        import torchaudio
        waveform, sr = torchaudio.load(tmp_in_path)
        data = waveform.numpy()
        if data.ndim == 2:
            data = data.mean(axis=0)
    finally:
        os.unlink(tmp_in_path)

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=-1) if data.shape[-1] <= data.shape[0] else data.mean(axis=0)

    # Resample to 16kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        import torchaudio
        waveform = torch.tensor(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        data = waveform.squeeze(0).numpy()

    return data


# ---------------------------------------------------------------------------
# Direct inference (bypasses lhotse dataloader)
# ---------------------------------------------------------------------------

def direct_transcribe(audio: np.ndarray) -> str:
    """Run full-file transcription using direct model forward pass."""
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(asr_model.device)
    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long).to(asr_model.device)

    with torch.no_grad():
        processed, processed_len = asr_model.preprocessor(
            input_signal=audio_tensor, length=audio_len,
        )
        encoded, encoded_len = asr_model.encoder(
            audio_signal=processed, length=processed_len,
        )
        hypotheses = asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=False,
        )
        return hypotheses[0].text


# ---------------------------------------------------------------------------
# Streaming transcription
# ---------------------------------------------------------------------------

def streaming_transcribe(audio: np.ndarray):
    """Yield incremental transcript deltas using conformer_stream_step."""
    model = asr_model
    device = model.device

    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        # Get mel spectrogram
        processed, processed_len = model.preprocessor(
            input_signal=audio_tensor, length=audio_len,
        )

        # Streaming config tells us chunk/shift sizes in frames
        scfg = model.encoder.streaming_cfg
        chunk_size = scfg.chunk_size
        shift_size = scfg.shift_size
        # chunk_size and shift_size may be lists (one per subsampling stage);
        # we need the input-level (first) value for slicing mel frames
        cs = chunk_size[0] if isinstance(chunk_size, (list, tuple)) else chunk_size
        ss = shift_size[0] if isinstance(shift_size, (list, tuple)) else shift_size
        pre_encode_cache = scfg.pre_encode_cache_size
        pre_cache = pre_encode_cache[0] if isinstance(pre_encode_cache, (list, tuple)) else pre_encode_cache

        total_frames = processed.shape[2]
        prev_text = ""
        previous_hypotheses = None

        # Initialize cache
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1)
        )

        # Pre-encode cache: pad left with zeros for first chunk
        if pre_cache > 0:
            pad = torch.zeros(
                processed.shape[0], processed.shape[1], pre_cache,
                device=device, dtype=processed.dtype,
            )
            processed = torch.cat([pad, processed], dim=2)
            total_frames = processed.shape[2]

        offset = 0
        while offset < total_frames:
            end = min(offset + cs, total_frames)
            chunk = processed[:, :, offset:end]
            chunk_len = torch.tensor([chunk.shape[2]], dtype=torch.long).to(device)

            result = model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                previous_hypotheses=previous_hypotheses,
                return_transcription=True,
            )

            # Unpack: (preds, texts_or_hyps, cache_ch, cache_t, cache_ch_len, best_hyp)
            (
                _greedy_preds,
                all_hyps,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                best_hyp,
            ) = result

            # Extract text from best hypothesis
            if best_hyp and len(best_hyp) > 0:
                hyp = best_hyp[0]
                current_text = hyp.text if hasattr(hyp, "text") else str(hyp)
            elif isinstance(all_hyps, list) and len(all_hyps) > 0:
                if isinstance(all_hyps[0], str):
                    current_text = all_hyps[0]
                elif hasattr(all_hyps[0], "text"):
                    current_text = all_hyps[0].text
                else:
                    current_text = str(all_hyps[0])
            else:
                current_text = ""

            # Carry forward hypotheses for RNNT
            previous_hypotheses = best_hyp

            if current_text and current_text != prev_text:
                delta = current_text[len(prev_text):]
                if delta:
                    yield delta
                prev_text = current_text

            offset += ss


async def sse_generator(audio: np.ndarray):
    """Generate SSE events from streaming transcription."""
    full_text = ""
    for delta in streaming_transcribe(audio):
        full_text += delta
        event = {"type": "transcript.text.delta", "delta": delta}
        yield f"data: {json.dumps(event)}\n\n"

    done_event = {"type": "transcript.text.done", "text": full_text.strip()}
    yield f"data: {json.dumps(done_event)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    response_format: Optional[str] = Form("json"),
    stream: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
):
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    is_stream = stream is not None and stream.lower() in ("true", "1", "yes")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio = load_audio(audio_bytes, file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")

    if is_stream:
        return StreamingResponse(
            sse_generator(audio),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming transcription
    try:
        text = direct_transcribe(audio)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    if response_format == "text":
        return JSONResponse(content=text, media_type="text/plain")
    elif response_format == "verbose_json":
        return JSONResponse(content={
            "text": text,
            "task": "transcribe",
            "language": "en",
            "duration": None,
        })
    else:
        return JSONResponse(content={"text": text})


@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia",
            }
        ],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": asr_model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeMo STT Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
