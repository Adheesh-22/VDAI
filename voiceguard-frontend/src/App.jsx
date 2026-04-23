import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioCapture } from './hooks/useAudioCapture';
import AudioControls from './components/AudioControls';
import BufferVisualization from './components/BufferVisualization';
import LFCCHeatmap from './components/LFCCHeatmap';
import StatsPanel from './components/StatsPanel';
import RawDataInspector from './components/RawDataInspector';
import './App.css';

const DEFAULT_WS_URL = 'ws://localhost:8000/ws';

function useChunksPerSec(chunkCount) {
  const prevRef = useRef({ count: 0, ts: Date.now() });
  const [cps, setCps] = useState(null);

  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const delta = chunkCount - prevRef.current.count;
      const elapsed = (now - prevRef.current.ts) / 1000;
      if (elapsed > 0) setCps(delta / elapsed);
      prevRef.current = { count: chunkCount, ts: now };
    }, 1000);
    return () => clearInterval(id);
  }, [chunkCount]);

  return cps;
}

export default function App() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [showSettings, setShowSettings] = useState(false);
  const [showValues, setShowValues] = useState(false);
  const [colorScheme, setColorScheme] = useState('BlueRed');
  const [lfccData, setLfccData] = useState(null);
  const [lfccShape, setLfccShape] = useState(null);
  const [bufferSamples, setBufferSamples] = useState(0);
  const [sessionLog, setSessionLog] = useState({ frames: [], predictions: [], meta: { started: null } });
  const bufferFloat32Ref = useRef(null);

  const { connected, lastMessage, retryCount, latencyMs, messageCount, send } = useWebSocket(wsUrl);

  const handleSend = useCallback((buffer) => {
    bufferFloat32Ref.current = new Float32Array(buffer);
    send(buffer);
  }, [send]);

  const { recording, paused, volume, waveform, chunkCount, error, startRecording, stopRecording, togglePause } =
    useAudioCapture(handleSend);

  const chunksPerSec = useChunksPerSec(chunkCount);

  useEffect(() => {
    if (!lastMessage) return;
    const { status, samples, lfcc_data, lfcc_shape, prediction, confidence } = lastMessage;

    if (typeof samples === 'number') setBufferSamples(samples);
    if (status === 'ready' || status === 'detection') setBufferSamples(lastMessage.buffer_samples ?? 24000);

    if (lfcc_data && Array.isArray(lfcc_data)) {
      setLfccData(lfcc_data);
      setLfccShape(lfcc_shape ?? null);
      setSessionLog((prev) => ({
        ...prev,
        frames: [...prev.frames.slice(-200), { ts: Date.now(), lfcc_shape }],
        meta: { ...prev.meta, started: prev.meta.started ?? Date.now() },
      }));
    }

    if (status === 'detection' && typeof prediction === 'number') {
      setSessionLog((prev) => ({
        ...prev,
        predictions: [...prev.predictions.slice(-500), { ts: Date.now(), prediction, confidence }],
      }));
    }
  }, [lastMessage]);

  const handleClearBuffer = useCallback(() => {
    setBufferSamples(0);
    setLfccData(null);
    setLfccShape(null);
  }, []);

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="header-left">
          <span className="logo-icon">🛡</span>
          <h1 className="app-title">VoiceGuard <span className="title-sub">Data Inspector</span></h1>
        </div>
        <div className="header-right">
          <div className={`conn-badge ${connected ? 'conn-ok' : 'conn-bad'}`}>
            {connected ? '● Connected' : retryCount > 0 ? `↺ Reconnecting (#${retryCount})` : '✗ Disconnected'}
          </div>
          <button
            id="btn-settings"
            className="btn btn-ghost"
            onClick={() => setShowSettings((v) => !v)}
          >
            ⚙ Settings
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="settings-bar">
          <label className="settings-label">
            WebSocket URL
            <input
              id="ws-url-input"
              className="settings-input"
              value={wsUrl}
              onChange={(e) => setWsUrl(e.target.value)}
            />
          </label>
          <label className="settings-label checkbox-label">
            <input
              id="toggle-show-values"
              type="checkbox"
              checked={showValues}
              onChange={(e) => setShowValues(e.target.checked)}
            />
            Show raw values on heatmap
          </label>
          <label className="settings-label">
            Heatmap colour scheme
            <select
              id="global-color-scheme"
              className="settings-select"
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value)}
            >
              <option value="BlueRed">BlueRed</option>
              <option value="Viridis">Viridis</option>
              <option value="Plasma">Plasma</option>
            </select>
          </label>
        </div>
      )}

      <main className="app-main">
        <div className="top-row">
          <AudioControls
            recording={recording}
            paused={paused}
            volume={volume}
            chunkCount={chunkCount}
            error={error}
            connected={connected}
            onStart={startRecording}
            onStop={stopRecording}
            onTogglePause={togglePause}
            onClearBuffer={handleClearBuffer}
          />
          <StatsPanel
            connected={connected}
            retryCount={retryCount}
            messageCount={messageCount}
            latencyMs={latencyMs}
            lastMessage={lastMessage}
            chunksPerSec={chunksPerSec}
          />
        </div>

        <div className="mid-row">
          <BufferVisualization waveform={waveform} bufferSamples={bufferSamples} />
          <LFCCHeatmap
            lfccData={lfccData}
            lfccShape={lfccShape}
            colorScheme={colorScheme}
            showValues={showValues}
          />
        </div>

        <div className="bottom-row">
          <RawDataInspector
            lastMessage={lastMessage}
            lfccData={lfccData}
            bufferFloat32={bufferFloat32Ref.current}
            sessionLog={sessionLog}
          />
        </div>
      </main>

      <footer className="app-footer">
        VoiceGuard · Sample rate: 16 000 Hz · Buffer: 24 000 samples · LFCC: 60 × N
      </footer>
    </div>
  );
}
