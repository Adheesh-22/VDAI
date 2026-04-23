import React, { useEffect, useRef } from 'react';

const BUFFER_SIZE = 24000;

export default function StatsPanel({
  connected,
  retryCount,
  messageCount,
  latencyMs,
  lastMessage,
  chunksPerSec,
}) {
  const statusColor = () => {
    if (!lastMessage) return '#aaa';
    if (lastMessage.status === 'detection') return '#16c79a';
    if (lastMessage.status === 'ready') return '#f7b731';
    if (lastMessage.status === 'buffering') return '#0f4c81';
    if (lastMessage.status === 'error') return '#ea5455';
    return '#aaa';
  };

  const renderStatusDetail = () => {
    if (!lastMessage) return <span className="stat-value dim">—</span>;
    const { status, samples, next_inference_in, prediction, confidence, buffer_samples } = lastMessage;
    const bufSize = buffer_samples ?? BUFFER_SIZE;

    if (status === 'buffering') {
      const pct = Math.min(100, Math.round((samples / bufSize) * 100));
      return (
        <span className="stat-value" style={{ color: '#0f4c81' }}>
          Buffering — {samples?.toLocaleString()} samples ({pct}%)
        </span>
      );
    }
    if (status === 'ready') {
      return (
        <span className="stat-value" style={{ color: '#f7b731' }}>
          Ready — next inference in {next_inference_in} chunk{next_inference_in !== 1 ? 's' : ''}
        </span>
      );
    }
    if (status === 'detection') {
      const pred = typeof prediction === 'number' ? (prediction * 100).toFixed(1) : '–';
      const conf = typeof confidence === 'number' ? (confidence * 100).toFixed(1) : '–';
      return (
        <span className="stat-value" style={{ color: '#16c79a' }}>
          Detection — score: {pred}% | confidence: {conf}%
        </span>
      );
    }
    if (status === 'error') {
      return <span className="stat-value" style={{ color: '#ea5455' }}>{lastMessage.message}</span>;
    }
    return <span className="stat-value dim">{status}</span>;
  };

  return (
    <div className="panel stats-panel">
      <div className="panel-header">
        <h2 className="panel-title">Real-time Stats</h2>
      </div>
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-label">Connection</span>
          {connected ? (
            <span className="stat-value" style={{ color: '#16c79a' }}>Connected ✓</span>
          ) : (
            <span className="stat-value" style={{ color: '#ea5455' }}>
              Disconnected ✗ {retryCount > 0 && `(retry #${retryCount})`}
            </span>
          )}
        </div>

        <div className="stat-card">
          <span className="stat-label">Messages received</span>
          <span className="stat-value">{messageCount.toLocaleString()}</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Round-trip latency</span>
          <span className="stat-value">
            {latencyMs !== null ? `${latencyMs} ms` : '—'}
          </span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Chunks / sec</span>
          <span className="stat-value">
            {chunksPerSec !== null ? chunksPerSec.toFixed(1) : '—'}
          </span>
        </div>

        <div className="stat-card stat-card-wide">
          <span className="stat-label">Backend status</span>
          <div className="status-dot-row">
            <span className="status-dot" style={{ background: statusColor() }} />
            {renderStatusDetail()}
          </div>
        </div>
      </div>
    </div>
  );
}
