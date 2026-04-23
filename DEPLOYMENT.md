# VoiceGuard-Proto — Deployment Guide

## Quick Start (Local Development)

```bash
# 1. Clone and enter the project
git clone https://github.com/your-org/voiceguard-proto.git
cd voiceguard-proto

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate a test model (skip if you have a real model.onnx)
python scripts/generate_dummy_model.py

# 5. Copy and configure environment
cp .env.example .env

# 6. Start the server
python main.py
# Server available at http://localhost:8000
```

---

## Environment Variables Reference

| Variable | Default | Type | Description |
|---|---|---|---|
| `MODEL_PATH` | `model.onnx` | string | Absolute or relative path to the ONNX classification model |
| `HOST` | `0.0.0.0` | string | Network interface to bind. Use `127.0.0.1` for local-only |
| `PORT` | `8000` | integer | TCP port the server listens on |
| `LOG_LEVEL` | `INFO` | enum | Python logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `RATE_LIMIT_ENABLED` | `true` | bool string | Set `false` to disable all rate limiting (useful for testing) |

---

## Docker Deployment

### docker-compose (recommended)

```bash
# Ensure model.onnx is in the project root
cp /path/to/your/model.onnx ./model.onnx

# Copy and edit environment file
cp .env.example .env

# Build and start
docker compose up --build -d

# Verify
curl http://localhost:8000/health

# View logs
docker compose logs -f voiceguard

# Stop
docker compose down
```

### Standalone Docker

```bash
# Build
docker build -t voiceguard-proto:latest .

# Run with model mounted
docker run -d \
  --name voiceguard \
  -p 8000:8000 \
  -v $(pwd)/model.onnx:/app/model.onnx:ro \
  -e LOG_LEVEL=INFO \
  -e RATE_LIMIT_ENABLED=true \
  --restart unless-stopped \
  voiceguard-proto:latest
```

---

## Production Deployment

### AWS (ECS + Fargate)

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name voiceguard-proto
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag voiceguard-proto:latest <account>.dkr.ecr.<region>.amazonaws.com/voiceguard-proto:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/voiceguard-proto:latest

# 2. Store model in S3
aws s3 cp model.onnx s3://your-bucket/models/model.onnx

# 3. Create ECS task definition
#    - Memory: 2048 MB minimum (librosa + ONNX runtime)
#    - CPU: 1024 units minimum
#    - Mount model via EFS volume or download from S3 at startup
#    - Set MODEL_PATH env var to /mnt/models/model.onnx

# 4. Create ECS service with Application Load Balancer
#    - Health check path: /health
#    - Health check interval: 30s
#    - Enable sticky sessions if using stateful WebSocket routing
```

### GCP (Cloud Run)

```bash
# 1. Push image to Artifact Registry
gcloud auth configure-docker
docker tag voiceguard-proto:latest gcr.io/YOUR_PROJECT/voiceguard-proto:latest
docker push gcr.io/YOUR_PROJECT/voiceguard-proto:latest

# 2. Deploy to Cloud Run
gcloud run deploy voiceguard-proto \
  --image gcr.io/YOUR_PROJECT/voiceguard-proto:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --port 8000 \
  --set-env-vars MODEL_PATH=/app/model.onnx,LOG_LEVEL=INFO
```

> **Note:** Cloud Run does not support persistent WebSocket connections natively — use GKE for production WebSocket workloads on GCP.

### Azure (Container Apps)

```bash
# 1. Create resource group and registry
az group create --name voiceguard-rg --location eastus
az acr create --resource-group voiceguard-rg --name voiceguardacr --sku Basic

# 2. Build and push
az acr build --registry voiceguardacr --image voiceguard-proto:latest .

# 3. Deploy Container App
az containerapp create \
  --name voiceguard-proto \
  --resource-group voiceguard-rg \
  --image voiceguardacr.azurecr.io/voiceguard-proto:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 1.0 \
  --memory 2.0Gi
```

---

## Monitoring Setup

### Prometheus Configuration

Add this scrape job to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: voiceguard-proto
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
```

### Key Metrics to Alert On

| Metric | Condition | Severity |
|---|---|---|
| `active_connections` | > 100 | Warning |
| `inference_duration_seconds` p99 | > 500ms | Warning |
| `errors_total{error_type="websocket_error"}` rate | > 5/min | Critical |
| `buffer_fills_total` rate | = 0 for 5min | Info |

### Grafana Dashboard (minimal)

```json
{
  "panels": [
    { "title": "Active Connections",   "expr": "active_connections" },
    { "title": "Inference Latency p99","expr": "histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))" },
    { "title": "Errors / min",         "expr": "rate(errors_total[1m]) * 60" },
    { "title": "Inferences / min",     "expr": "rate(websocket_connections_total[1m]) * 60" }
  ]
}
```

---

## Performance Tuning

### Uvicorn Workers

For CPU-bound workloads, run multiple workers behind a process manager:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note:** `AudioProcessor` is instantiated per WebSocket connection, so worker isolation is safe.

### ONNX Runtime Optimisations

```python
# In AudioProcessor.__init__, replace InferenceSession with:
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
```

### Memory

- Each `AudioProcessor` holds a 24 000-element float32 buffer ≈ **96 KB**.
- With 100 concurrent connections: ≈ **9.6 MB** of buffer RAM.
- ORT session loading is the dominant cost (~50–200 MB depending on model size) — consider sharing a single session across connections using a module-level singleton.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Model not loaded` on every WS connection | `model.onnx` missing or path wrong | Verify `MODEL_PATH` and file exists; check `GET /health` |
| `Invalid byte length N: not aligned` | Client sending non-float32 data | Ensure frontend encodes audio as `Float32Array` before sending |
| `429 Too Many Requests` on `/health` | Rate limit hit | Set `RATE_LIMIT_ENABLED=false` in `.env` for testing |
| High inference latency (>200ms) | ONNX model too large or no SIMD | Use `onnxruntime-gpu` or reduce model size; check CPU SIMD support |
| WebSocket closes immediately | Unhandled exception in loop | Check `LOG_LEVEL=DEBUG` logs; ensure librosa version ≥ 0.10 |
| `active_connections` gauge drifts up | Exception bypassing `finally` | This is handled; if observed, check for `SystemExit` signals |

---

## Security Hardening

- [ ] Set `HOST=127.0.0.1` and terminate TLS at a reverse proxy (nginx/Caddy)
- [ ] Replace `allow_origins=["*"]` with an explicit list of trusted origins
- [ ] Set `RATE_LIMIT_ENABLED=true` in all production deployments
- [ ] Store `MODEL_PATH` in a secrets manager (AWS Secrets Manager / GCP Secret Manager)
- [ ] Run the container as a non-root user: add `USER 1001` to the Dockerfile
- [ ] Restrict container capabilities: `--cap-drop ALL` in docker run
- [ ] Enable HTTPS-only WebSocket (`wss://`) at the load balancer layer
- [ ] Regularly rotate Docker base images to pick up OS security patches
- [ ] Pin exact dependency versions in `requirements.txt` before production release
- [ ] Scan image with `docker scout` or `trivy` before deploying
