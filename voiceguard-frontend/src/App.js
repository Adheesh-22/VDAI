import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const BUFFER_SIZE = 24000;
const CHUNK_SIZE = 4096;
const SAMPLE_RATE = 16000;
const WS_DEFAULT = 'ws://127.0.0.1:8000/ws';

function blueRedColor(v) {
  const n = Math.max(0, Math.min(1, (v + 1) / 2));
  return `rgb(${Math.round(n * 220)},20,${Math.round((1 - n) * 220)})`;
}
function viridisColor(v) {
  const n = Math.max(0, Math.min(1, (v + 1) / 2));
  return `rgb(${Math.min(255,Math.round((0.267+0.733*n)*220))},${Math.min(255,Math.round((0.1+0.8*n)*220))},${Math.min(255,Math.round((0.4+0.2*n)*220))})`;
}
function getColor(scheme, v) { return scheme === 'Viridis' ? viridisColor(v) : blueRedColor(v); }

function normalizeMatrix(matrix) {
  let lo = Infinity, hi = -Infinity;
  for (const row of matrix) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = hi - lo || 1;
  return { norm: matrix.map(r => r.map(v => ((v - lo) / range) * 2 - 1)), lo, hi };
}

function encodeWAV(f32, sr) {
  const n = f32.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0,'RIFF'); v.setUint32(4,36+n*2,true); ws(8,'WAVE'); ws(12,'fmt ');
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,sr,true); v.setUint32(28,sr*2,true); v.setUint16(32,2,true);
  v.setUint16(34,16,true); ws(36,'data'); v.setUint32(40,n*2,true);
  for (let i = 0; i < n; i++) { const s = Math.max(-1,Math.min(1,f32[i])); v.setInt16(44+i*2,s<0?s*0x8000:s*0x7FFF,true); }
  return buf;
}

function triggerDownload(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(a.href);
}

export default function App() {
  // WebSocket
  const wsRef = useRef(null);
  const retryRef = useRef(0);
  const retryTimer = useRef(null);
  const sentAt = useRef(null);
  const [wsUrl] = useState(WS_DEFAULT);
  const [connected, setConnected] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [msgCount, setMsgCount] = useState(0);
  const [latency, setLatency] = useState(null);
  const [lastMsg, setLastMsg] = useState(null);

  // Audio
  const audioCtxRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const pausedRef = useRef(false);
  const chunkCountRef = useRef(0);
  const bufferAudio = useRef(null);
  const [recording, setRecording] = useState(false);
  const [paused, setPaused] = useState(false);
  const [micVolume, setMicVolume] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [micError, setMicError] = useState('');

  // Live data FROM backend
  const [bufferSamples, setBufferSamples] = useState(0);
  const [bufferPct, setBufferPct] = useState(0);
  const [backendRms, setBackendRms] = useState(0);
  const [backendWaveform, setBackendWaveform] = useState([]);
  const [lfccData, setLfccData] = useState(null);
  const [lfccShape, setLfccShape] = useState(null);
  const [modelAvailable, setModelAvailable] = useState(false);
  const [sessionLog, setSessionLog] = useState({ frames: [], startedAt: null });

  // UI
  const [colorScheme, setColorScheme] = useState('BlueRed');
  const [showSettings, setShowSettings] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  // Chunks/sec
  const [cps, setCps] = useState(null);
  const prevCPS = useRef({ count: 0, ts: Date.now() });
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const delta = chunkCountRef.current - prevCPS.current.count;
      const dt = (now - prevCPS.current.ts) / 1000;
      if (dt > 0) setCps(delta / dt);
      prevCPS.current = { count: chunkCountRef.current, ts: now };
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // WebSocket connect
  const connectWS = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState < 2) return;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => { setConnected(true); retryRef.current = 0; setRetryCount(0); };

    ws.onmessage = (e) => {
      if (sentAt.current) {
        setLatency(Math.round(performance.now() - sentAt.current));
        sentAt.current = null;
      }
      try {
        const data = JSON.parse(e.data);
        setLastMsg(data);
        setMsgCount(c => c + 1);

        // Update live metrics from every message
        if (typeof data.buffer_pct === 'number') setBufferPct(data.buffer_pct);
        if (typeof data.samples === 'number') setBufferSamples(data.samples);
        if (data.buffer_pct >= 100) setBufferSamples(data.buffer_size ?? BUFFER_SIZE);
        if (typeof data.rms === 'number') setBackendRms(data.rms);
        if (Array.isArray(data.waveform) && data.waveform.length > 0) setBackendWaveform(data.waveform);
        if (typeof data.model_available === 'boolean') setModelAvailable(data.model_available);

        if (Array.isArray(data.lfcc_data) && data.lfcc_data.length > 0) {
          setLfccData(data.lfcc_data);
          setLfccShape(data.lfcc_shape ?? null);
          setSessionLog(prev => ({
            frames: [...prev.frames.slice(-200), { ts: Date.now(), shape: data.lfcc_shape }],
            startedAt: prev.startedAt ?? Date.now(),
          }));
        }
      } catch { /* non-JSON */ }
    };

    ws.onclose = () => {
      setConnected(false); wsRef.current = null;
      const delay = Math.min(3000 * 2 ** retryRef.current, 30000);
      retryRef.current++; setRetryCount(retryRef.current);
      retryTimer.current = setTimeout(connectWS, delay);
    };
    ws.onerror = () => ws.close();
  }, [wsUrl]);

  useEffect(() => {
    connectWS();
    return () => { clearTimeout(retryTimer.current); wsRef.current?.close(); };
  }, [connectWS]);

  const sendAudio = useCallback((buf) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sentAt.current = performance.now();
      wsRef.current.send(buf);
    }
  }, []);

  // Microphone
  const startRecording = useCallback(async () => {
    setMicError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const proc = ctx.createScriptProcessor(CHUNK_SIZE, 1, 1);
      processorRef.current = proc;

      proc.onaudioprocess = (e) => {
        const raw = e.inputBuffer.getChannelData(0);
        bufferAudio.current = new Float32Array(raw);
        let sum = 0;
        for (let i = 0; i < raw.length; i++) sum += raw[i] ** 2;
        setMicVolume(Math.min(1, Math.sqrt(sum / raw.length) * 8));
        if (!pausedRef.current) {
          sendAudio(bufferAudio.current.buffer.slice(0));
          chunkCountRef.current++;
          setChunkCount(chunkCountRef.current);
        }
      };

      source.connect(proc); proc.connect(ctx.destination);
      setRecording(true); pausedRef.current = false; setPaused(false);
    } catch (err) {
      setMicError(err.name === 'NotAllowedError'
        ? 'Microphone access denied. Click the 🔒 lock in the address bar → Microphone → Allow, then reload.'
        : `Audio error: ${err.message}. Try Chrome or Firefox.`);
    }
  }, [sendAudio]);

  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect(); sourceRef.current?.disconnect();
    audioCtxRef.current?.close(); streamRef.current?.getTracks().forEach(t => t.stop());
    processorRef.current = sourceRef.current = audioCtxRef.current = streamRef.current = null;
    chunkCountRef.current = 0;
    setRecording(false); setPaused(false); setMicVolume(0); setChunkCount(0);
  }, []);

  const togglePause = useCallback(() => {
    pausedRef.current = !pausedRef.current; setPaused(pausedRef.current);
  }, []);

  const clearBuffer = useCallback(() => {
    setBufferSamples(0); setBufferPct(0); setLfccData(null); setLfccShape(null);
    setBackendWaveform([]); setBackendRms(0);
  }, []);

  // Downloads
  const downloadCSV = () => {
    if (!lfccData) return;
    triggerDownload(new Blob([lfccData.map(r => r.join(',')).join('\n')], { type: 'text/csv' }), `lfcc_${Date.now()}.csv`);
  };
  const downloadWAV = () => {
    if (!bufferAudio.current) return;
    triggerDownload(new Blob([encodeWAV(bufferAudio.current, SAMPLE_RATE)], { type: 'audio/wav' }), `buf_${Date.now()}.wav`);
  };
  const downloadSession = () => {
    triggerDownload(new Blob([JSON.stringify(sessionLog, null, 2)], { type: 'application/json' }), `session_${Date.now()}.json`);
  };

  // Waveform canvas — uses backend waveform when available
  const waveRef = useRef(null);
  useEffect(() => {
    const canvas = waveRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 600, H = 100;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0d0d1a'; ctx.fillRect(0, 0, W, H);
    const data = backendWaveform;
    if (!data || data.length === 0) return;
    const step = W / data.length, mid = H / 2;
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#0f4c81'); grad.addColorStop(0.5, '#16c79a'); grad.addColorStop(1, '#0f4c81');
    ctx.strokeStyle = grad; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = i * step, y = mid + data[i] * mid * 0.9;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
  }, [backendWaveform]);

  // LFCC heatmap canvas
  const heatmapRef = useRef(null);
  useEffect(() => {
    const canvas = heatmapRef.current;
    if (!canvas || !lfccData || lfccData.length === 0) return;
    const rows = lfccData.length, cols = lfccData[0].length;
    const { norm } = normalizeMatrix(lfccData);
    const cw = Math.max(2, Math.floor(canvas.offsetWidth / cols));
    const ch = Math.max(2, Math.floor(200 / rows));
    canvas.width = cw * cols; canvas.height = ch * rows;
    const ctx = canvas.getContext('2d');
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = getColor(colorScheme, norm[r][c]);
        ctx.fillRect(c * cw, r * ch, cw, ch);
      }
  }, [lfccData, colorScheme]);

  // VU meter
  const BARS = 20;
  const filled = Math.round(micVolume * BARS);
  const barColor = bufferPct >= 100 ? '#16c79a' : bufferPct >= 30 ? '#f7b731' : '#ea5455';

  // Status text
  const statusColor = () => {
    if (!lastMsg) return '#7878a0';
    return { detection:'#16c79a', features_ready:'#16c79a', ready:'#f7b731', buffering:'#1a6fbd', connected:'#7878a0', error:'#ea5455' }[lastMsg.status] ?? '#7878a0';
  };
  const statusText = () => {
    if (!lastMsg) return 'Waiting for backend…';
    const { status, samples, next_inference_in, buffer_size, buffer_pct } = lastMsg;
    const bsz = buffer_size ?? BUFFER_SIZE;
    if (status === 'connected') return `Backend connected — model ${modelAvailable ? 'loaded ✓' : 'not loaded (feature-only mode)'}`;
    if (status === 'buffering') return `Filling buffer — ${samples?.toLocaleString()} / ${bsz.toLocaleString()} samples (${buffer_pct ?? 0}%)`;
    if (status === 'ready') return `Buffer full — LFCC extraction in ${next_inference_in} chunk${next_inference_in !== 1 ? 's' : ''}`;
    if (status === 'features_ready') return `LFCC extracted ✓ — ${lfccShape ? lfccShape[0]+'×'+lfccShape[1] : ''} — no model yet`;
    if (status === 'detection') return `Inference complete — raw score: ${lastMsg.prediction?.toFixed(4)}`;
    if (status === 'error') return `Error: ${lastMsg.message}`;
    return status;
  };

  return (
    <div className="root">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <span className="logo-glyph">🛡</span>
          <div>
            <h1 className="site-title">VoiceGuard</h1>
            <p className="site-sub">Live Audio Feature Inspector · 16 kHz · 24 000-sample buffer</p>
          </div>
        </div>
        <div className="header-right">
          <span className={`conn-pill ${connected ? 'conn-ok' : 'conn-bad'}`}>
            {connected ? '● Connected' : retryCount > 0 ? `↺ Retry #${retryCount}` : '✗ Disconnected'}
          </span>
          <span className={`model-pill ${modelAvailable ? 'model-ok' : 'model-no'}`}>
            {modelAvailable ? '🧠 Model loaded' : '⚙ Feature-only mode'}
          </span>
          <button className="btn-ghost" onClick={() => setShowSettings(v => !v)}>⚙ Settings</button>
        </div>
      </header>

      {/* Settings */}
      {showSettings && (
        <div className="settings-bar">
          <label>Heatmap colour
            <select className="settings-select" value={colorScheme} onChange={e => setColorScheme(e.target.value)}>
              <option value="BlueRed">BlueRed</option>
              <option value="Viridis">Viridis</option>
            </select>
          </label>
          <span className="settings-note">WebSocket: {wsUrl}</span>
        </div>
      )}

      <main className="main">

        {/* ROW 1: Controls + Stats */}
        <div className="row two-col">

          {/* Audio Controls */}
          <section className="panel">
            <h2 className="panel-title"><span className={`rec-dot ${recording && !paused ? 'dot-rec' : ''}`} /> Audio Input</h2>
            <span className="badge-hz">16 000 Hz · 4 096-sample chunks</span>

            {micError && <div className="error-box">{micError}</div>}

            <div className="btn-row">
              {!recording ? (
                <button className="btn btn-start" onClick={startRecording}>
                  🎙 Start Recording
                </button>
              ) : (
                <>
                  <button className="btn btn-stop" onClick={stopRecording}>■ Stop</button>
                  <button className={`btn ${paused ? 'btn-resume' : 'btn-pause'}`} onClick={togglePause}>
                    {paused ? '▶ Resume' : '⏸ Pause'}
                  </button>
                  <button className="btn btn-ghost-sm" onClick={clearBuffer}>↺ Clear Buffer</button>
                </>
              )}
            </div>

            <div className="vu-row">
              <span className="vu-label">Mic level</span>
              <div className="vu-track">
                {Array.from({length: BARS}).map((_, i) => (
                  <div key={i} className={`vu-bar ${i < filled ? (i < 14 ? 'bar-g' : i < 18 ? 'bar-y' : 'bar-r') : ''}`} />
                ))}
              </div>
              <span className="vu-pct">{Math.round(micVolume * 100)}%</span>
            </div>

            <div className="vu-row">
              <span className="vu-label">RMS</span>
              <div className="vu-track">
                {Array.from({length: BARS}).map((_, i) => {
                  const rmsNorm = Math.min(1, backendRms * 50);
                  const rFilled = Math.round(rmsNorm * BARS);
                  return <div key={i} className={`vu-bar ${i < rFilled ? 'bar-g' : ''}`} />;
                })}
              </div>
              <span className="vu-pct">{backendRms.toFixed(4)}</span>
            </div>

            <div className="chip-row">
              <span className="chip">Chunks sent: {chunkCount.toLocaleString()}</span>
              <span className="chip">{paused ? '⏸ Paused' : recording ? '● Streaming' : '○ Idle'}</span>
            </div>
          </section>

          {/* Stats */}
          <section className="panel">
            <h2 className="panel-title">Real-time Stats</h2>
            <div className="stats-grid">
              <div className="stat-box">
                <span className="stat-label">WebSocket</span>
                <span className="stat-val" style={{color: connected ? '#16c79a' : '#ea5455'}}>
                  {connected ? 'Connected ✓' : `Disconnected${retryCount > 0 ? ` (retry #${retryCount})` : ''}`}
                </span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Messages received</span>
                <span className="stat-val">{msgCount.toLocaleString()}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Round-trip latency</span>
                <span className="stat-val">{latency !== null ? `${latency} ms` : '—'}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Chunks / sec</span>
                <span className="stat-val">{cps !== null ? cps.toFixed(1) : '—'}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Buffer RMS energy</span>
                <span className="stat-val">{backendRms > 0 ? backendRms.toFixed(5) : '—'}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">LFCC shape</span>
                <span className="stat-val">{lfccShape ? `${lfccShape[0]} × ${lfccShape[1]}` : '—'}</span>
              </div>
              <div className="stat-box stat-wide">
                <span className="stat-label">Backend status</span>
                <div className="status-row">
                  <span className="status-dot" style={{background: statusColor()}} />
                  <span className="stat-val" style={{color: statusColor(), fontSize: 12}}>{statusText()}</span>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* ROW 2: Waveform + LFCC */}
        <div className="row two-col">

          {/* Buffer waveform */}
          <section className="panel">
            <h2 className="panel-title">Buffer Waveform</h2>
            <div className="progress-wrap">
              <div className="prog-track">
                <div className="prog-fill" style={{width: `${bufferPct}%`, background: barColor}} />
              </div>
              <span className="prog-label" style={{color: barColor}}>
                {bufferPct.toFixed(1)}%  ({bufferSamples.toLocaleString()} / {BUFFER_SIZE.toLocaleString()} samples)
              </span>
            </div>
            <canvas ref={waveRef} className="wave-canvas" />
            <div className="canvas-sub">
              Backend rolling buffer — {BUFFER_SIZE / SAMPLE_RATE}s window
              {backendWaveform.length === 0 && !recording && <span> — start recording to see data</span>}
            </div>
          </section>

          {/* LFCC Heatmap */}
          <section className="panel">
            <div className="hm-header">
              <h2 className="panel-title">LFCC Feature Matrix</h2>
              {lfccShape && <span className="badge-hz">{lfccShape[0]} × {lfccShape[1]}</span>}
            </div>
            {lfccData ? (
              <>
                <div className="hm-axis-labels">
                  <span>← Time frames (0…{(lfccData[0]?.length ?? 1) - 1}) →</span>
                  <span>↕ Coefficients C0…C{lfccData.length - 1}</span>
                </div>
                <div className="hm-wrap">
                  <canvas ref={heatmapRef} className="hm-canvas" />
                </div>
                <div className="hm-legend">
                  <span style={{color: getColor(colorScheme, -1)}}>■ Low</span>
                  <span>LFCC amplitude</span>
                  <span style={{color: getColor(colorScheme, 1)}}>■ High</span>
                </div>
              </>
            ) : (
              <div className="hm-empty">
                <div className="hm-empty-icon">⬜</div>
                <p>Waiting for buffer to fill…</p>
                <p className="hm-empty-sub">
                  {bufferPct > 0 ? `${bufferPct.toFixed(0)}% filled — keep recording` : 'Start recording to populate the buffer'}
                </p>
                <div className="hm-progress-mini">
                  <div className="prog-track"><div className="prog-fill" style={{width:`${bufferPct}%`,background:barColor}}/></div>
                </div>
              </div>
            )}
          </section>
        </div>

        {/* ROW 3: Inspector */}
        <div className="row">
          <section className="panel inspector">
            <button className="inspector-toggle" onClick={() => setInspectorOpen(v => !v)}>
              <span>{inspectorOpen ? '▲' : '▼'}  Raw Data Inspector</span>
              {lastMsg && <span className="badge-status">{lastMsg.status}</span>}
            </button>
            {inspectorOpen && (
              <div className="inspector-body">
                <div className="inspect-cols">
                  <div className="inspect-col">
                    <h3 className="inspect-h">Last WebSocket Message (lfcc_data truncated)</h3>
                    <pre className="json-pre">
{lastMsg
  ? JSON.stringify({...lastMsg, lfcc_data: lastMsg.lfcc_data ? `[${lastMsg.lfcc_data.length} rows × ${lastMsg.lfcc_data[0]?.length} cols]` : undefined}, null, 2)
  : 'No messages yet'}
                    </pre>
                  </div>
                  <div className="inspect-col">
                    <h3 className="inspect-h">LFCC Preview — rows 0–4, cols 0–9</h3>
                    <pre className="json-pre">
{lfccData
  ? JSON.stringify(lfccData.slice(0,5).map(r => r.slice(0,10).map(v => +v.toFixed(4))), null, 2)
  : 'No LFCC data yet — buffer must fill first'}
                    </pre>
                    <h3 className="inspect-h" style={{marginTop:12}}>Last Audio Chunk — first 20 samples</h3>
                    <pre className="json-pre">
{bufferAudio.current
  ? JSON.stringify(Array.from(bufferAudio.current.slice(0,20)).map(v => +v.toFixed(6)), null, 2)
  : 'No audio captured yet'}
                    </pre>
                  </div>
                </div>
                <div className="export-row">
                  <button className="btn btn-export" onClick={downloadCSV} disabled={!lfccData}>⬇ LFCC as CSV</button>
                  <button className="btn btn-export" onClick={downloadWAV} disabled={!bufferAudio.current}>⬇ Buffer as WAV</button>
                  <button className="btn btn-export" onClick={downloadSession}>⬇ Session JSON</button>
                </div>
              </div>
            )}
          </section>
        </div>

      </main>
      <footer className="footer">
        VoiceGuard · Feature-extraction mode · Model inference will be added when model.onnx is available
      </footer>
    </div>
  );
}
