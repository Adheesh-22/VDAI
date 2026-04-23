# VoiceGuard — Data Meaning Reference

This document explains every parameter measured, extracted, and displayed by the VoiceGuard system.

---

## 1. Audio Input Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **Sample Rate** | 16 000 Hz | The microphone captures 16 000 audio samples every second. This rate is standard for speech processing — it captures frequencies up to 8 000 Hz (Nyquist theorem), which covers the full range of the human voice. |
| **Chunk Size** | 4 096 samples | Audio is split into chunks of 4 096 samples (~256 ms each) before being sent to the backend over WebSocket. This is the smallest unit of data transmitted. |
| **Float32 Format** | 32-bit float | Each audio sample is a decimal number between -1.0 and +1.0. Negative = compression phase of the sound wave, positive = rarefaction phase. |
| **Bit Depth** | 32-bit | Precision of each sample value. Higher bit depth → more dynamic range → finer detail in quiet sounds. |

---

## 2. Buffer Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **Buffer Size** | 24 000 samples | The backend holds a rolling window of the last 24 000 samples (1.5 seconds of audio). New chunks overwrite the oldest samples. This is the data that goes into LFCC extraction. |
| **Buffer Fill %** | 0–100% | How full the 1.5s buffer is. At 0% the buffer is silent zeros. At 100% it contains 1.5 seconds of real audio. LFCC extraction only runs when it reaches 100%. |
| **Samples count** | 0–24 000 | Raw count of non-zero samples in the buffer. Zeros at the start mean the buffer is not yet full. |
| **RMS (Root Mean Square)** | 0.0 – ~0.5 | The energy level of the audio in the buffer. Calculated as `sqrt(mean(sample²))`. Higher RMS = louder sound. Silence = RMS near 0. Normal speech = 0.01–0.1. Loud speech = 0.1–0.5. |

---

## 3. Waveform Display

| Parameter | Meaning |
|-----------|---------|
| **Waveform** | A visual plot of the audio signal amplitude over time. The Y-axis shows sample value (−1 to +1). The X-axis shows time. Peaks and valleys represent sound waves. Flat line = silence. |
| **Waveform Snapshot** | The backend sends 128 evenly-spaced points sampled from the 24 000-sample buffer so the frontend can draw it efficiently without transferring all 24 000 values. |
| **Centre line** | The zero-crossing line. When audio oscillates around this line symmetrically, it means the microphone is well-calibrated. |

---

## 4. LFCC — Linear Frequency Cepstral Coefficients

This is the most important feature extracted from the audio.

### What is LFCC?

LFCC is a compact mathematical representation of the **spectral envelope** (tonal shape) of a sound. It captures *how* the voice sounds — its texture, timbre, and phonetic content — without caring about volume or pitch.

Deepfake detection models use LFCC because real and synthetic voices have different spectral envelopes even when saying identical words.

### How it is computed (step by step)

```
Raw audio buffer (24 000 samples)
        ↓
STFT (Short-Time Fourier Transform, n_fft=512, hop=256)
   → Converts time-domain audio to frequency-domain
   → Produces a spectrogram: how much of each frequency exists at each point in time
        ↓
Power Spectrum  =  |STFT|²
   → Squares the magnitudes to get power (energy) at each frequency
        ↓
Log scale  →  librosa.power_to_db()
   → Converts power to decibels (log scale matches how human hearing works)
        ↓
DCT (Discrete Cosine Transform, type=2, axis=0)
   → Compresses the log-power spectrum into a compact set of coefficients
   → Similar to how JPEG compresses images — keeps the important info, discards redundancy
        ↓
Take first 60 rows  →  LFCC matrix (60 × ~93)
```

### LFCC Matrix Dimensions

| Dimension | Size | Meaning |
|-----------|------|---------|
| **Rows (coefficients)** | 60 | Each row is one LFCC coefficient. C0 = overall energy. C1–C59 = increasingly fine spectral details. Lower coefficients carry more information. |
| **Columns (frames)** | ~93 | Each column is one time frame (≈ 16 ms of audio). The exact number depends on buffer size and hop length: `floor((24000 - 512) / 256) + 1 ≈ 93`. |
| **Total values** | 60 × 93 = 5 580 | Every LFCC matrix has 5 580 float32 values representing 1.5 seconds of audio. |

### LFCC Heatmap Interpretation

| Colour | Meaning |
|--------|---------|
| **Dark blue** (BlueRed scheme) | Low coefficient value — little energy at that frequency band / time frame |
| **Dark red** (BlueRed scheme) | High coefficient value — strong energy at that frequency band / time frame |
| **Horizontal bands** | Consistent tonal character across time (stable vowel sounds look like horizontal stripes) |
| **Vertical stripes** | Sudden changes (consonants, stops, transitions between phonemes) |
| **Noisy scattered pattern** | Background noise or unvoiced sounds (s, f, sh) |

---

## 5. WebSocket Communication Parameters

| Parameter | Meaning |
|-----------|---------|
| **Messages received** | Count of JSON messages sent by the backend since connection opened. One message per audio chunk received. |
| **Round-trip latency** | Time in milliseconds from when the frontend sends an audio chunk to when the backend's response arrives. Measures network + processing delay. Typical: 5–50 ms on localhost. |
| **Chunks / sec** | How many 4096-sample audio chunks are being sent per second. At 16 000 Hz this should be approximately `16000 / 4096 ≈ 3.9 chunks/sec`. |

---

## 6. Backend Status Values

| Status | Meaning |
|--------|---------|
| `connected` | WebSocket just opened. Backend reports whether a model is loaded. |
| `buffering` | Buffer is not yet full. Shows current sample count. Audio is being collected but LFCC extraction has not started. |
| `ready` | Buffer is full. Backend is waiting for the inference interval counter to tick. Shows how many more chunks before LFCC extraction runs. |
| `features_ready` | Buffer is full AND LFCC has been extracted — but **no model is loaded yet**. The LFCC matrix is sent to the frontend for visualisation. This is the current operating mode. |
| `detection` | A model IS loaded and inference has run. The model's raw output score is shown. |
| `error` | Something went wrong (invalid audio data, rate limit exceeded, etc.). |

---

## 7. What "Feature-only mode" means

Right now, the system runs in **feature-extraction-only mode**:

```
Microphone → Buffer → LFCC extraction → Frontend visualisation
```

When the model is trained and `model.onnx` is placed in the backend folder, the pipeline becomes:

```
Microphone → Buffer → LFCC extraction → ONNX model inference → Prediction score
```

The frontend already handles both modes — it will automatically switch to showing inference results when the model is available.

---

## 8. What the Model Will Receive

When the model is integrated, it receives:

| Input | Shape | Description |
|-------|-------|-------------|
| LFCC tensor | `(1, 1, 60, 93)` | Batch=1, Channels=1, Height=60 coefficients, Width=93 time frames |

The extra dimensions `(1, 1, ...)` are added by `prepare_input()` to match the format expected by a 2D convolutional neural network.

---

## 9. Microphone Level vs Buffer RMS

These are two different measurements:

| Metric | Source | What it measures |
|--------|--------|-----------------|
| **Mic Level (VU meter)** | Frontend (browser) | Instantaneous loudness of the current 4096-sample chunk being recorded. Updates ~4 times/sec. |
| **Buffer RMS** | Backend (Python) | Root-mean-square energy across the entire 24 000-sample rolling buffer. More stable and representative of recent speech energy overall. |

---

## 10. Export Formats

| Export | Format | Contents |
|--------|--------|----------|
| **LFCC CSV** | Comma-separated values | 60 rows × N columns of float values. Each row = one coefficient. Each column = one time frame. Open in Excel or NumPy. |
| **Buffer WAV** | PCM 16-bit, mono, 16 kHz | The last audio chunk captured from the microphone encoded as a standard WAV file. Can be played back or loaded into audio tools. |
| **Session JSON** | JSON | Timestamps and LFCC shapes for every extraction event in the current session. Useful for tracking how many full-buffer cycles occurred. |
