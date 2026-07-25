FROM python:3.11-slim

# espeak-ng is required by kokoro-onnx's phonemizer (espeakng-loader uses it as fallback)
RUN apt-get update && apt-get install -y espeak-ng && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (no torch — kokoro-onnx uses onnxruntime instead)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model file locations (INT8 ONNX ~88 MB, FP32 ~310 MB)
ENV MODEL_PATH=/app/models/kokoro-v1.0.int8.onnx
ENV VOICES_PATH=/app/models/voices-v1.0.bin

# Download model files and run synthesis smoke-test at build time
COPY warmup.py .
RUN python warmup.py

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
