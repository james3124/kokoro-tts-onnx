# kokoro-tts

Self-hosted Kokoro-82M TTS FastAPI service — deployed on Render, audio stored in Supabase Storage.

> **v4.0 — ONNX migration:** The inference engine was replaced from the original PyTorch `KPipeline` to [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) with the **INT8 quantised model**. The API is fully backwards-compatible — no changes required in n8n or any existing client.

## Stack

| Layer | Tech |
|---|---|
| TTS engine | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — INT8 ONNX, CPU inference |
| Model | Kokoro-82M `kokoro-v1.0.int8.onnx` (88 MB) |
| API | FastAPI + Uvicorn |
| Hosting | Render (free tier compatible) |
| Storage | Supabase Storage |

---

## RAM comparison

| | v3 (PyTorch) | v4 (ONNX INT8) |
|---|---|---|
| Model on disk | ~350 MB (HF cache) | **88 MB** |
| Model in RAM | ~600–700 MB | **~150–200 MB** |
| Framework overhead | ~300 MB (torch) | **0** |
| Cold-start | 20–30 s | **3–5 s** |

The INT8 model fits comfortably in Render's free-tier 512 MB RAM limit.

---

## Deploy to Render

### 1. Push to GitHub
```bash
git init && git add . && git commit -m "init"
git remote add origin https://github.com/YOU/kokoro-tts
git push -u origin main
```

### 2. Create Render Web Service
- **New → Web Service** → connect repo
- Render auto-detects `render.yaml`
- The build step runs `warmup.py`, which downloads the ONNX model and voices files (~90 MB total) and verifies synthesis before the service goes live.

### 3. Set these in Render Dashboard → Environment
| Key | Value |
|---|---|
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_KEY` | your service role key |
| `DEFAULT_VOICE` | e.g. `af_heart` (optional) |
| `LANG_CODE` | e.g. `a` for American EN (optional) |

`MODEL_PATH`, `VOICES_PATH`, and `ORT_THREADS` are pre-set in `render.yaml` and do not need to be changed unless you switch model variants.

### 4. Supabase bucket
Create a bucket named **`audio`** → set to **Public**.

---

## API

### `POST /tts`
```json
{
  "text": "Hello from Kokoro!",
  "voice": "af_heart",
  "speed": 1.0,
  "lang_code": "a",
  "split_pattern": "\n+"
}
```
All fields except `text` are optional — fall back to env var defaults.

**Response:**
```json
{
  "url": "https://xxxx.supabase.co/storage/v1/object/public/audio/tts_uuid.wav",
  "filename": "tts_uuid.wav",
  "duration_seconds": 1.84,
  "voice": "af_heart",
  "lang_code": "a"
}
```

### `GET /health`
```json
{
  "ok": true,
  "model_loaded": true,
  "engine": "kokoro-onnx (INT8 ONNX)",
  "model_path": "./models/kokoro-v1.0.int8.onnx",
  "idle_seconds": 12,
  "unloads_after_seconds": 60,
  "default_voice": "af_heart",
  "default_lang": "a"
}
```

### `GET /voices`
Returns all voice IDs grouped by accent/gender.

### `GET /languages`
Returns all supported language codes.

---

## Voices

| ID | Gender | Accent |
|---|---|---|
| `af_heart` ⭐ | F | American |
| `af_bella` | F | American |
| `af_nicole` | F | American |
| `af_sarah` | F | American |
| `af_sky` | F | American |
| `am_adam` | M | American |
| `am_michael` | M | American |
| `bf_emma` | F | British |
| `bf_isabella` | F | British |
| `bm_george` | M | British |
| `bm_lewis` | M | British |

## Languages

| Code | Language | Notes |
|---|---|---|
| `a` | 🇺🇸 American English | — |
| `b` | 🇬🇧 British English | — |
| `e` | 🇪🇸 Spanish | — |
| `f` | 🇫🇷 French | — |
| `h` | 🇮🇳 Hindi | — |
| `i` | 🇮🇹 Italian | — |
| `j` | 🇯🇵 Japanese | espeak fallback; quality is acceptable |
| `p` | 🇧🇷 Brazilian Portuguese | — |
| `z` | 🇨🇳 Mandarin Chinese | espeak fallback; the v1.1-zh model gives better results |

Language codes are mapped internally to espeak-ng strings (`a` → `en-us`, `b` → `en-gb`, etc.) — your existing requests do not change.

---

## Model variants

The default is the INT8 model. To switch, update `MODEL_PATH` in `render.yaml` or your `.env`:

| Variant | File | Size | Notes |
|---|---|---|---|
| **INT8** (default) | `kokoro-v1.0.int8.onnx` | 88 MB | Best for free tier |
| FP16 | `kokoro-v1.0.fp16.onnx` | 169 MB | Higher quality |
| FP32 | `kokoro-v1.0.onnx` | 310 MB | Maximum quality |

All variants are available at the [kokoro-onnx releases page](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0).

---

## ONNX session tuning

| Env var | Default | Description |
|---|---|---|
| `ORT_THREADS` | `1` | ONNX intra-op threads. Increase if your host has spare cores. |
| `MODEL_PATH` | `./models/kokoro-v1.0.int8.onnx` | Path to the `.onnx` model file |
| `VOICES_PATH` | `./models/voices-v1.0.bin` | Path to the voices binary |
| `UNLOAD_AFTER_SECONDS` | `60` | Seconds of idle before the model is unloaded from RAM |

---

## n8n Integration

No changes needed from v3. Same request, same response:

```
Method: POST
URL: https://kokoro-tts.onrender.com/tts
Body:
{
  "text": "{{ $json.script }}",
  "voice": "af_heart",
  "speed": 1.0
}
```
Output `{{ $json.url }}` → CutEngine audio track URL.

---

## Phoneme Override Syntax

For custom pronunciation, Kokoro supports inline phoneme hints:
```
[Kokoro](/kˈOkəɹO/) is a TTS model.
```

---

## Local Dev

```bash
sudo apt-get install -y espeak-ng
pip install -r requirements.txt

# Download model files (first time only)
python warmup.py

cp .env.example .env   # fill in Supabase creds
uvicorn main:app --reload
```

Test:
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_heart"}'
```
