import React, { useRef, useEffect, useMemo } from 'react';

const BUFFER_SIZE = 24000;
const CANVAS_W = 600;
const CANVAS_H = 100;

export default function BufferVisualization({ waveform, bufferSamples }) {
  const canvasRef = useRef(null);
  const samples = bufferSamples ?? 0;
  const pct = Math.min(1, samples / BUFFER_SIZE);
  const pctDisplay = Math.round(pct * 100);

  const barColor = useMemo(() => {
    if (pct >= 1.0) return '#16c79a';
    if (pct >= 0.3) return '#f7b731';
    return '#ea5455';
  }, [pct]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_W * dpr;
    canvas.height = CANVAS_H * dpr;
    canvas.style.width = `${CANVAS_W}px`;
    canvas.style.height = `${CANVAS_H}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    if (!waveform || waveform.length === 0) return;

    const mid = CANVAS_H / 2;
    const step = CANVAS_W / waveform.length;

    const grad = ctx.createLinearGradient(0, 0, CANVAS_W, 0);
    grad.addColorStop(0, '#0f4c81');
    grad.addColorStop(0.5, '#16c79a');
    grad.addColorStop(1, '#0f4c81');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    for (let i = 0; i < waveform.length; i++) {
      const x = i * step;
      const y = mid + waveform[i] * mid * 0.9;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(CANVAS_W, mid);
    ctx.stroke();
  }, [waveform]);

  return (
    <div className="panel buffer-panel">
      <div className="panel-header">
        <h2 className="panel-title">Buffer Status</h2>
        <span className="info-chip">1.5 s window</span>
      </div>

      <div className="progress-wrap">
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${pctDisplay}%`, background: barColor }}
          />
        </div>
        <span className="progress-label" style={{ color: barColor }}>
          {pctDisplay}% ({samples.toLocaleString()} / {BUFFER_SIZE.toLocaleString()} samples)
        </span>
      </div>

      <canvas ref={canvasRef} className="waveform-canvas" />
      <p className="canvas-subtitle">Rolling buffer waveform</p>
    </div>
  );
}
