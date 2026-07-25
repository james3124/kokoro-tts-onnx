"""
warmup.py — runs once at build time (Dockerfile / render.yaml buildCommand).

What it does:
  1. Downloads the INT8 ONNX model (~88 MB) and voices file if not already present.
  2. Loads an ONNX session with the same session options used in main.py.
  3. Synthesises a short phrase to verify everything is working.

Env vars:
  MODEL_PATH   path where the model will be stored (default ./models/kokoro-v1.0.int8.onnx)
  VOICES_PATH  path where voices will be stored   (default ./models/voices-v1.0.bin)
  ORT_THREADS  ONNX intra-op threads              (default 1)
"""

import gc
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL   = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

MODEL_PATH  = os.environ.get("MODEL_PATH",  "./models/kokoro-v1.0.int8.onnx")
VOICES_PATH = os.environ.get("VOICES_PATH", "./models/voices-v1.0.bin")
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))


def _download(url: str, dest: str) -> None:
    """Download a file only if it does not already exist."""
    path = Path(dest)
    if path.exists():
        size_mb = path.stat().st_size / 1_048_576
        print(f"  ✔  Already exists: {dest}  ({size_mb:.1f} MB)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓  Downloading {url}")
    print(f"     → {dest}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            print(f"\r     {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress
        size_mb = Path(dest).stat().st_size / 1_048_576
        print(f"     ✔  Saved ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"\n  ✗  Download failed: {exc}", file=sys.stderr)
        raise


# ── 1. Download model files ────────────────────────────────────────────────────
print("=== kokoro-onnx warmup ===")
print(f"Model  : {MODEL_PATH}")
print(f"Voices : {VOICES_PATH}")
print()

_download(MODEL_URL,  MODEL_PATH)
_download(VOICES_URL, VOICES_PATH)
print()

# ── 2. Build ONNX session ──────────────────────────────────────────────────────
print("Loading ONNX session...")
import onnxruntime as rt
from kokoro_onnx import Kokoro

sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_mem_pattern       = True
sess_options.enable_cpu_mem_arena     = True
sess_options.intra_op_num_threads     = ORT_THREADS
sess_options.execution_mode           = rt.ExecutionMode.ORT_SEQUENTIAL

session = rt.InferenceSession(
    MODEL_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
kokoro = Kokoro.from_session(session, VOICES_PATH)
print("  ✔  Session ready")

# ── 3. Synthesis smoke-test ────────────────────────────────────────────────────
print("Running synthesis smoke-test...")
audio, sr = kokoro.create("Ready.", voice="af_heart", speed=1.0, lang="en-us")
duration  = len(audio) / sr
print(f"  ✔  Generated {duration:.2f}s of audio at {sr} Hz")

# Free memory — main.py will reload lazily on first real request
del kokoro, session
gc.collect()

print()
print("=== Warmup complete ===")
