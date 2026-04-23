import React, { useState } from 'react';
import { downloadCSV, downloadBufferWAV, downloadJSON } from '../utils/exportHelpers';

export default function RawDataInspector({ lastMessage, lfccData, bufferFloat32, sessionLog }) {
  const [open, setOpen] = useState(false);

  const handleDownloadCSV = () => {
    if (!lfccData || lfccData.length === 0) return;
    downloadCSV(lfccData, `lfcc_${Date.now()}.csv`);
  };

  const handleDownloadWAV = () => {
    if (!bufferFloat32 || bufferFloat32.length === 0) return;
    downloadBufferWAV(bufferFloat32, 16000, `buffer_${Date.now()}.wav`);
  };

  const handleDownloadSession = () => {
    downloadJSON(sessionLog, `session_${Date.now()}.json`);
  };

  const previewLFCC = lfccData
    ? lfccData.slice(0, 5).map((row) => row.slice(0, 10))
    : null;

  return (
    <div className="panel inspector-panel">
      <button
        id="btn-toggle-inspector"
        className="inspector-toggle"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{open ? '▲' : '▼'} Raw Data Inspector</span>
        {lastMessage && (
          <span className="inspector-badge">{lastMessage.status}</span>
        )}
      </button>

      {open && (
        <div className="inspector-body">
          <section className="inspector-section">
            <h3>Last WebSocket Message</h3>
            <pre className="json-viewer">
              {lastMessage
                ? JSON.stringify(
                    {
                      ...lastMessage,
                      lfcc_data: lastMessage.lfcc_data
                        ? `[…${lastMessage.lfcc_data.length} rows × ${lastMessage.lfcc_data[0]?.length} cols…]`
                        : undefined,
                    },
                    null,
                    2
                  )
                : 'No messages yet'}
            </pre>
          </section>

          <section className="inspector-section">
            <h3>LFCC Preview (first 5 rows × 10 cols)</h3>
            <pre className="json-viewer">
              {previewLFCC
                ? JSON.stringify(previewLFCC.map((r) => r.map((v) => +v.toFixed(4))), null, 2)
                : 'No LFCC data yet'}
            </pre>
          </section>

          <section className="inspector-section">
            <h3>Buffer Audio Preview (first 20 samples)</h3>
            <pre className="json-viewer">
              {bufferFloat32 && bufferFloat32.length > 0
                ? JSON.stringify(
                    Array.from(bufferFloat32.slice(0, 20)).map((v) => +v.toFixed(6)),
                    null,
                    2
                  )
                : 'No buffer data yet'}
            </pre>
          </section>

          <div className="export-row">
            <button
              id="btn-download-lfcc"
              className="btn btn-secondary"
              onClick={handleDownloadCSV}
              disabled={!lfccData || lfccData.length === 0}
            >
              ⬇ LFCC CSV
            </button>
            <button
              id="btn-download-wav"
              className="btn btn-secondary"
              onClick={handleDownloadWAV}
              disabled={!bufferFloat32 || bufferFloat32.length === 0}
            >
              ⬇ Buffer WAV
            </button>
            <button
              id="btn-download-session"
              className="btn btn-secondary"
              onClick={handleDownloadSession}
            >
              ⬇ Session JSON
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
