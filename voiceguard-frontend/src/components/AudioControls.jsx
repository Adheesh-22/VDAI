import React, { useCallback, useState } from 'react';

export default function AudioControls({
  recording,
  paused,
  volume,
  chunkCount,
  error,
  connected,
  onStart,
  onStop,
  onTogglePause,
  onClearBuffer,
}) {
  const [showMicHelp, setShowMicHelp] = useState(false);

  const handleStart = useCallback(() => {
    if (!connected) return;
    onStart();
  }, [connected, onStart]);

  const vuBars = 20;
  const filledBars = Math.round(volume * vuBars);

  return (
    <div className="panel audio-controls-panel">
      <div className="panel-header">
        <h2 className="panel-title">
          <span className={`rec-dot ${recording && !paused ? 'recording' : ''}`} />
          Audio Input
        </h2>
        <span className="sample-rate-badge">16 000 Hz</span>
      </div>

      {error === 'microphone_denied' && (
        <div className="error-banner">
          <span>🎤 Microphone access denied.</span>
          <button className="link-btn" onClick={() => setShowMicHelp((v) => !v)}>
            How to fix?
          </button>
          {showMicHelp && (
            <div className="mic-help">
              Go to your browser's address bar → click the 🔒 lock icon → Microphone → Allow.
              Then reload the page.
            </div>
          )}
        </div>
      )}

      {error && error !== 'microphone_denied' && (
        <div className="error-banner">⚠ {error}</div>
      )}

      <div className="controls-row">
        {!recording ? (
          <button
            id="btn-start-recording"
            className="btn btn-primary"
            onClick={handleStart}
            disabled={!connected}
          >
            {connected ? '▶ Start Recording' : 'Waiting for backend…'}
          </button>
        ) : (
          <>
            <button id="btn-stop-recording" className="btn btn-danger" onClick={onStop}>
              ■ Stop
            </button>
            <button
              id="btn-pause-recording"
              className={`btn ${paused ? 'btn-success' : 'btn-warning'}`}
              onClick={onTogglePause}
            >
              {paused ? '▶ Resume' : '⏸ Pause'}
            </button>
            <button id="btn-clear-buffer" className="btn btn-secondary" onClick={onClearBuffer}>
              ↺ Clear Buffer
            </button>
          </>
        )}
      </div>

      <div className="vu-row">
        <span className="vu-label">Level</span>
        <div className="vu-meter">
          {Array.from({ length: vuBars }).map((_, i) => (
            <div
              key={i}
              className={`vu-bar ${i < filledBars ? (i < 14 ? 'bar-green' : i < 18 ? 'bar-yellow' : 'bar-red') : ''}`}
            />
          ))}
        </div>
        <span className="vu-pct">{Math.round(volume * 100)}%</span>
      </div>

      <div className="info-row">
        <span className="info-chip">Chunks sent: {chunkCount.toLocaleString()}</span>
        <span className="info-chip">
          {paused ? '⏸ Paused' : recording ? '● Streaming' : '○ Idle'}
        </span>
      </div>
    </div>
  );
}
