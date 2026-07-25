import os, gc, re, time
import io
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "audio")
DEFAULT_VOICE   = os.environ.get("DEFAULT_VOICE", "af_heart")
DEFAULT_LANG    = os.environ.get("LANG_CODE", "a")
SAMPLE_RATE     = 24000
UNLOAD_AFTER    = int(os.environ.get("UNLOAD_AFTER_SECONDS", "60"))

# Model file paths — downloaded by warmup.py at build time
MODEL_PATH  = os.environ.get("MODEL_PATH",  "./models/kokoro-v1.0.int8.onnx")
VOICES_PATH = os.environ.get("VOICES_PATH", "./models/voices-v1.0.bin")

# ONNX runtime threads (1 is safest for free-tier single-core hosts)
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))

# ── Lang code mapping ──────────────────────────────────────────────────────────
# Maps the original single-char API lang codes → espeak-ng language strings
# used internally by kokoro-onnx's phonemizer.
LANG_CODE_MAP = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "e": "es",     # Spanish
    "f": "fr-fr",  # French
    "h": "hi",     # Hindi
    "i": "it",     # Italian
    "j": "ja",     # Japanese  (espeak fallback; misaki[ja] gives better quality)
    "p": "pt-br",  # Brazilian Portuguese
    "z": "cmn",    # Mandarin  (espeak fallback; requires v1.1-zh model for best quality)
}

LANG_MAP = {
    "a": "American English", "b": "British English",
    "e": "Spanish",          "f": "French",
    "h": "Hindi",            "i": "Italian",
    "j": "Japanese",         "p": "Brazilian Portuguese",
    "z": "Mandarin",
}

VOICES = {
    "american_female": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky"],
    "american_male":   ["am_adam", "am_michael"],
    "british_female":  ["bf_emma", "bf_isabella"],
    "british_male":    ["bm_george", "bm_lewis"],
}
ALL_VOICES = [v for group in VOICES.values() for v in group]

# ── State ──────────────────────────────────────────────────────────────────────
_kokoro        = None   # single kokoro_onnx.Kokoro instance (replaces _pipelines dict)
_last_request  = 0.0
_supabase      = None


# ── Kokoro ONNX helpers ────────────────────────────────────────────────────────
def get_kokoro():
    """
    Lazily load and return a single reusable Kokoro ONNX session.
    Uses the INT8 model for minimum RAM (~88 MB) and CPU-optimised session options.
    """
    global _kokoro
    if _kokoro is None:
        logger.info(f"Loading kokoro-onnx model from '{MODEL_PATH}' ...")
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        sess_options = rt.SessionOptions()
        # Enable all graph optimisations (fused ops, const folding, etc.)
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable memory arenas on free tier — releases memory immediately after inference
        # instead of pooling it, which prevents OOM on 512 MB hosts
        sess_options.enable_mem_pattern   = False
        sess_options.enable_cpu_mem_arena = False
        # Single-thread inference: avoids contention on free-tier single-core hosts
        sess_options.intra_op_num_threads = ORT_THREADS
        sess_options.execution_mode       = rt.ExecutionMode.ORT_SEQUENTIAL

        session = rt.InferenceSession(
            MODEL_PATH,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Reuse the pre-created session rather than letting Kokoro() open a new one
        _kokoro = Kokoro.from_session(session, VOICES_PATH)
        gc.collect()
        logger.info("✅ kokoro-onnx ready (INT8, CPU, single session)")
    return _kokoro


def unload_all():
    global _kokoro
    if _kokoro is not None:
        _kokoro = None
        gc.collect()
        logger.info("♻️  Model unloaded — RAM freed")


async def unload_watcher():
    while True:
        await asyncio.sleep(30)
        if _kokoro is not None and time.time() - _last_request > UNLOAD_AFTER:
            unload_all()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _supabase
    logger.info("Connecting to Supabase...")
    _supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )
    logger.info(f"🚀 Ready — model loads on first request, unloads after {UNLOAD_AFTER}s idle")
    task = asyncio.create_task(unload_watcher())
    yield
    task.cancel()
    unload_all()


app = FastAPI(title="Kokoro TTS", version="4.1.0", lifespan=lifespan)


# ── Schemas ────────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: Optional[str] = Field(default=None)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lang_code: Optional[str] = Field(default=None)
    split_pattern: Optional[str] = Field(default=r"\n+")


class TTSResponse(BaseModel):
    url: str
    filename: str
    duration_seconds: float
    voice: str
    lang_code: str


# ── Core ───────────────────────────────────────────────────────────────────────
def _synthesize(text: str, voice: str, speed: float, lang_code: str, split_pattern: Optional[str]) -> np.ndarray:
    """
    Synthesise audio using kokoro-onnx.

    split_pattern behaviour mirrors the original KPipeline implementation:
    the text is pre-split on the regex pattern and each segment is synthesised
    separately, then concatenated. kokoro-onnx additionally handles internal
    phoneme-length chunking automatically within each segment.
    """
    kokoro = get_kokoro()
    lang   = LANG_CODE_MAP[lang_code]

    # Pre-split on caller's pattern (default "\n+") exactly as KPipeline did
    if split_pattern:
        parts = [p.strip() for p in re.split(split_pattern, text) if p.strip()]
    else:
        parts = [text]

    chunks = []
    for part in parts:
        audio, _ = kokoro.create(part, voice=voice, speed=speed, lang=lang)
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("No audio chunks returned")
    result = np.concatenate(chunks)
    del chunks      # free intermediate buffers immediately
    gc.collect()
    return result


def _to_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _upload(filename: str, wav: bytes) -> str:
    _supabase.storage.from_(SUPABASE_BUCKET).upload(
        path=filename, file=wav,
        file_options={"content-type": "audio/wav"},
    )
    return _supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "kokoro-tts", "version": "4.1.0"}


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    global _last_request
    _last_request = time.time()

    voice     = req.voice     or DEFAULT_VOICE
    lang_code = req.lang_code or DEFAULT_LANG

    if voice not in ALL_VOICES:
        raise HTTPException(400, f"Unknown voice '{voice}'")
    if lang_code not in LANG_MAP:
        raise HTTPException(400, f"Unknown lang_code '{lang_code}'")

    loop = asyncio.get_event_loop()

    try:
        audio = await loop.run_in_executor(
            None, lambda: _synthesize(req.text, voice, req.speed, lang_code, req.split_pattern)
        )
    except Exception as e:
        logger.exception("Synthesis failed")
        raise HTTPException(500, f"TTS error: {e}")

    wav      = _to_wav(audio)
    duration = round(len(audio) / SAMPLE_RATE, 2)
    del audio       # free numpy array before upload — reduces peak RAM
    gc.collect()

    filename = f"tts_{uuid.uuid4()}.wav"

    try:
        url = await loop.run_in_executor(None, lambda: _upload(filename, wav))
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, f"Storage error: {e}")

    _last_request = time.time()
    logger.info(f"✅ {filename} | {duration}s | {voice}")

    return TTSResponse(
        url=url, filename=filename,
        duration_seconds=duration, voice=voice, lang_code=lang_code,
    )


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": _kokoro is not None,
        "engine": "kokoro-onnx (INT8 ONNX)",
        "model_path": MODEL_PATH,
        "idle_seconds": round(time.time() - _last_request) if _last_request else None,
        "unloads_after_seconds": UNLOAD_AFTER,
        "default_voice": DEFAULT_VOICE,
        "default_lang": DEFAULT_LANG,
    }


@app.get("/voices")
async def voices():
    return {"default": DEFAULT_VOICE, "all": ALL_VOICES, "grouped": VOICES}


@app.get("/languages")
async def languages():
    return {"default": DEFAULT_LANG, "available": LANG_MAP}
