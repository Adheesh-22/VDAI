# VoiceGuard-Proto

A real-time deepfake voice detection backend. Audio streams from a browser microphone over WebSocket, is processed through an LFCC feature extraction pipeline, and classified by an ONNX model — entirely in RAM with zero disk I/O.

**Documentation**
- [Deployment Guide](DEPLOYMENT.md) — local dev, Docker, AWS/GCP/Azure, monitoring, security
- [Architecture](ARCHITECTURE.md) — system diagram, data flow, scaling guide

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust values as needed. All variables have defaults and are optional.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.onnx` | Path to the ONNX classification model |
| `HOST` | `0.0.0.0` | Network interface the server binds to |
| `PORT` | `8000` | TCP port the server listens on |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

---

## Quick Start

```bash
python main.py
```

Override environment variables inline:

```bash
MODEL_PATH=/models/voiceguard.onnx PORT=9000 LOG_LEVEL=DEBUG python main.py
```

---

## API Endpoints

### `GET /health`

Returns server and model status.

**Response**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### `GET /stats`

Returns cumulative inference performance metrics across all WebSocket sessions since server start.

**Response**
```json
{
  "total_inferences": 42,
  "avg_inference_ms": 18.347
}
```

---

### `WebSocket /ws`

Real-time audio classification endpoint.

#### Input Format

- **Transport**: Binary WebSocket frames
- **Encoding**: Raw IEEE 754 float32 PCM, little-endian
- **Chunk size**: 4096 samples (0.256 seconds at 16 000 Hz)
- **Sample rate**: 16 000 Hz

#### Output JSON Schemas

**Buffering** — sent while the 24 000-sample rolling buffer is filling:
```json
{ "status": "buffering", "samples": 18432 }
```

**Ready** — buffer full, inference not due yet (throttled to every 4 chunks):
```json
{ "status": "ready", "next_inference_in": 3 }
```

**Detection** — inference result (emitted every 4th chunk once buffer is full):
```json
{ "status": "detection", "prediction": 0.91, "confidence": 0.09 }
```

**Error — model not loaded** — sent immediately after connection, then connection closes:
```json
{ "status": "error", "message": "Model not loaded" }
```

---

## Processing Pipeline

```
Binary PCM bytes (4096 × float32)
        │
        ▼
NumPy rolling buffer (24 000 samples)
        │
        ▼ (every 4th chunk)
librosa.stft  →  power spectrum  →  librosa.power_to_db
        │
        ▼
scipy DCT (type=2, axis=0, norm='ortho')  →  first 60 coefficients
        │
        ▼
Shape: (60, ~93)  →  expand to (1, 1, 60, ~93)
        │
        ▼
ONNX Runtime InferenceSession
        │
        ▼
{"prediction": float, "confidence": float}
```
