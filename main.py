import os
import time
import logging
import numpy as np
import json
import uvicorn
import librosa
import onnxruntime as ort
from scipy.fftpack import dct
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

limiter = Limiter(key_func=get_remote_address, enabled=RATE_LIMIT_ENABLED)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUFFER_SIZE = 24000
CHUNK_SIZE = 4096
INFERENCE_INTERVAL = 4
WAVEFORM_POINTS = 128

_global_stats = {
    "total_inferences": 0,
    "total_inference_time": 0.0,
}

websocket_connections_total = Counter(
    "websocket_connections_total",
    "Total WebSocket connections accepted",
)
influence_duration_seconds = Histogram(
    "inference_duration_seconds",
    "ONNX inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
buffer_fills_total = Counter(
    "buffer_fills_total",
    "Total times the rolling buffer reached full capacity",
)
errors_total = Counter(
    "errors_total",
    "Total errors by type",
    ["error_type"],
)
active_connections = Gauge(
    "active_connections",
    "Current number of active WebSocket connections",
)


class AudioProcessor:
    def __init__(self):
        self.buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.samples_received = 0
        self.inference_counter = 0
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        self.session = None
        self.model_loaded = False
        self._try_load_model()

    def _try_load_model(self):
        try:
            self.session = ort.InferenceSession(MODEL_PATH)
            self.model_loaded = True
            logging.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            self.session = None
            self.model_loaded = False
            logging.warning(f"No model loaded ({e}) — running in feature-extraction-only mode")

    def validate_chunk(self, raw_bytes: bytes) -> tuple:
        if len(raw_bytes) == 0:
            return False, "Received empty chunk"
        if len(raw_bytes) % 4 != 0:
            return False, f"Invalid byte length {len(raw_bytes)}: must be a multiple of 4 (float32)"
        return True, None

    def process_chunk(self, raw_bytes: bytes):
        is_valid, msg = self.validate_chunk(raw_bytes)
        if not is_valid:
            raise ValueError(msg)
        chunk = np.frombuffer(raw_bytes, dtype=np.float32)
        self.buffer = np.roll(self.buffer, -len(chunk))
        self.buffer[-len(chunk):] = chunk
        self.samples_received += len(chunk)

    def reset_buffer(self):
        self.buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.samples_received = 0
        self.inference_counter = 0
        logging.info("Buffer reset")

    def count_nonzero(self) -> int:
        return int(np.count_nonzero(self.buffer))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.buffer ** 2)))

    def waveform_snapshot(self) -> list:
        step = max(1, BUFFER_SIZE // WAVEFORM_POINTS)
        return [float(self.buffer[i * step]) for i in range(WAVEFORM_POINTS)]

    def extract_lfcc(self) -> np.ndarray:
        stft = librosa.stft(self.buffer, n_fft=512, hop_length=256, center=False)
        power_spec = np.abs(stft) ** 2
        db_spec = librosa.power_to_db(power_spec)
        lfcc_full = dct(db_spec, type=2, axis=0, norm='ortho')
        return lfcc_full[:60, :]

    def prepare_input(self, lfcc: np.ndarray) -> np.ndarray:
        tensor = np.expand_dims(lfcc, axis=0)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor.astype(np.float32)

    def run_inference(self, input_tensor: np.ndarray):
        input_name = self.session.get_inputs()[0].name
        t_start = time.time()
        output = self.session.run(None, {input_name: input_tensor})
        elapsed = time.time() - t_start
        self.total_inferences += 1
        self.total_inference_time += elapsed
        _global_stats["total_inferences"] = self.total_inferences
        _global_stats["total_inference_time"] = self.total_inference_time
        return output, elapsed


@app.get("/health")
@limiter.limit("30/minute")
def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": any(
            True for _ in [None] if os.path.exists(MODEL_PATH)
        ),
        "rate_limiting_enabled": RATE_LIMIT_ENABLED,
    }


@app.get("/stats")
@limiter.limit("20/minute")
def get_stats(request: Request):
    total = _global_stats["total_inferences"]
    total_time = _global_stats["total_inference_time"]
    avg_ms = (total_time / total * 1000) if total > 0 else 0
    return {
        "total_inferences": total,
        "avg_inference_ms": round(avg_ms, 3),
    }


@app.get("/metrics")
@limiter.limit("60/minute")
def metrics(request: Request):
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections_total.inc()
    active_connections.inc()
    logging.info("WebSocket connection accepted")

    processor = AudioProcessor()
    buffer_filled = False

    # Tell client whether model is available upfront
    await websocket.send_text(json.dumps({
        "status": "connected",
        "model_available": processor.model_loaded,
        "buffer_size": BUFFER_SIZE,
        "sample_rate": 16000,
    }))

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()

            try:
                processor.process_chunk(raw_bytes)
            except ValueError as e:
                logging.error(f"Invalid chunk: {e}")
                errors_total.labels(error_type="invalid_chunk").inc()
                await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
                processor.reset_buffer()
                buffer_filled = False
                continue

            current_samples = min(processor.samples_received, BUFFER_SIZE)
            rms_val = processor.rms()
            waveform = processor.waveform_snapshot()
            buf_pct = round((current_samples / BUFFER_SIZE) * 100, 1)

            if current_samples < BUFFER_SIZE:
                # ── Still filling the buffer ──────────────────────────────
                await websocket.send_text(json.dumps({
                    "status": "buffering",
                    "samples": current_samples,
                    "buffer_size": BUFFER_SIZE,
                    "buffer_pct": buf_pct,
                    "rms": round(rms_val, 6),
                    "waveform": waveform,
                }))

            else:
                # ── Buffer full ───────────────────────────────────────────
                if not buffer_filled:
                    buffer_fills_total.inc()
                    buffer_filled = True

                processor.inference_counter += 1

                if processor.inference_counter % INFERENCE_INTERVAL == 0:
                    lfcc = processor.extract_lfcc()

                    if processor.model_loaded:
                        # Run real inference
                        input_tensor = processor.prepare_input(lfcc)
                        output, elapsed = processor.run_inference(input_tensor)
                        influence_duration_seconds.observe(elapsed)
                        logging.info(f"Inference: prediction={output[0][0]:.4f} time={elapsed*1000:.1f}ms")
                        await websocket.send_text(json.dumps({
                            "status": "detection",
                            "prediction": float(output[0][0]),
                            "confidence": float(output[0][1]),
                            "lfcc_shape": list(lfcc.shape),
                            "lfcc_data": lfcc.tolist(),
                            "buffer_size": BUFFER_SIZE,
                            "buffer_pct": 100.0,
                            "rms": round(rms_val, 6),
                            "waveform": waveform,
                        }))
                    else:
                        # No model yet — send LFCC data anyway for visualisation
                        await websocket.send_text(json.dumps({
                            "status": "features_ready",
                            "lfcc_shape": list(lfcc.shape),
                            "lfcc_data": lfcc.tolist(),
                            "buffer_size": BUFFER_SIZE,
                            "buffer_pct": 100.0,
                            "rms": round(rms_val, 6),
                            "waveform": waveform,
                            "model_available": False,
                        }))
                else:
                    chunks_remaining = INFERENCE_INTERVAL - (processor.inference_counter % INFERENCE_INTERVAL)
                    await websocket.send_text(json.dumps({
                        "status": "ready",
                        "next_inference_in": chunks_remaining,
                        "buffer_size": BUFFER_SIZE,
                        "buffer_pct": 100.0,
                        "rms": round(rms_val, 6),
                        "waveform": waveform,
                    }))

    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        errors_total.labels(error_type="websocket_error").inc()
    finally:
        active_connections.dec()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
