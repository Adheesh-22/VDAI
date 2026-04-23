# VoiceGuard-Proto — Architecture

## System Architecture

```
                          Browser / Client
                               │
                    PCM Float32 binary frames
                    (4096 samples / 0.256s)
                               │
                        ┌──────▼──────┐
                        │  FastAPI    │
                        │  :8000      │
                        │             │
              ┌─────────┤  /ws        │
              │         │  /health    ├───────► Prometheus
              │         │  /stats     │         :9090
              │         │  /metrics   │
              │         └─────────────┘
              │                │
              │        ┌───────▼────────┐
              │        │ AudioProcessor  │
              │        │                │
              │        │ Rolling Buffer  │
              │        │ float32[24000] │
              │        │ (1.5s @ 16kHz) │
              │        └───────┬────────┘
              │                │  buffer full
              │        ┌───────▼────────┐
              │        │ LFCC Extraction │
              │        │                │
              │        │ librosa.stft   │
              │        │ power_to_db    │
              │        │ scipy DCT      │
              │        │ → (60, ~93)    │
              │        └───────┬────────┘
              │                │  every 4th chunk
              │        ┌───────▼────────┐
              │        │ ONNX Runtime   │
              │        │                │
              │        │ model.onnx     │
              │        │ → [pred, conf] │
              │        └───────┬────────┘
              │                │
              └────────────────┘
                  JSON response
```

---

## Data Flow

```
1. CLIENT  →  sends raw Float32 PCM bytes  →  SERVER /ws
2. SERVER  →  validate_chunk()             →  reject misaligned / empty
3. SERVER  →  process_chunk()             →  np.roll + insert into buffer
4. SERVER  →  count_nonzero()             →  < 24000 → {"status":"buffering"}
5. SERVER  →  inference_counter += 1
              counter % 4 != 0            →  {"status":"ready"}
              counter % 4 == 0            →  run pipeline:
                  extract_lfcc()          →  STFT → power → dB → DCT → [:60,:]
                  prepare_input()         →  expand to (1,1,60,T)
                  run_inference()         →  ORT session.run()
                                          →  {"status":"detection", ...}
```

---

## Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| **FastAPI app** | `main.py` | HTTP/WS routing, CORS, rate limiting, Prometheus |
| **AudioProcessor** | `main.py` | Buffer management, LFCC extraction, ORT inference, stats |
| **Limiter** | `main.py` | Per-IP rate limiting via slowapi |
| **Prometheus metrics** | `main.py` | Counters, histograms, gauges exposed at `/metrics` |
| **ONNX model** | `model.onnx` | Binary classifier: real voice vs deepfake |
| **Docker** | `Dockerfile`, `docker-compose.yml` | Containerisation and model volume mounting |
| **CI/CD** | `.github/workflows/ci.yml` | Test matrix, lint, Docker build + push |
| **Tests** | `test_main.py`, `test_websocket.py` | Unit + integration coverage |
| **Benchmark** | `benchmark.py` | Standalone pipeline latency profiling |

---

## Scaling Considerations

### Vertical Scaling

- Increasing CPU cores primarily benefits ORT inference (BLAS threading).
- Set `intra_op_num_threads` in `SessionOptions` to match available cores.
- Memory per connection is small (~96 KB buffer); the dominant cost is the ORT session (~50–200 MB).

### Horizontal Scaling

- Each `AudioProcessor` is **connection-local** (no shared mutable state per connection).
- `_global_stats` and Prometheus metrics are **process-local** — aggregate across replicas at the Prometheus layer using `sum()`.
- The rolling buffer is stateful per connection — a client that reconnects to a different replica starts a fresh buffer. Use sticky sessions on the load balancer to avoid this.

### Model Sharing

- Currently, one `ort.InferenceSession` is created **per WebSocket connection** — this is safe but memory-inefficient at scale.
- For high-concurrency deployments, refactor to a **module-level singleton** session shared across connections (ORT sessions are thread-safe for `run()`).

```python
# Module-level singleton pattern
_shared_session = None

def get_session():
    global _shared_session
    if _shared_session is None:
        _shared_session = ort.InferenceSession(MODEL_PATH)
    return _shared_session
```

### Inference Throttling

- `INFERENCE_INTERVAL = 4` (every ~1 second) keeps CPU load bounded.
- Reduce to `2` for lower latency, increase to `8` for reduced CPU at the cost of responsiveness.
