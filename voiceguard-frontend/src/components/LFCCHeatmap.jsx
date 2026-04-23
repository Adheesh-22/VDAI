import React, { useMemo, useState } from 'react';
import HeatMap from 'react-heatmap-grid';

const COLOR_SCHEMES = {
  BlueRed: (v) => {
    const r = Math.round(Math.max(0, v) * 255);
    const b = Math.round(Math.max(0, -v) * 255);
    return `rgb(${r},0,${b})`;
  },
  Viridis: (v) => {
    const n = (v + 1) / 2;
    const r = Math.round((0.267 + 0.004 * n + 0.329 * n ** 3) * 255);
    const g = Math.round((0.004 + 0.873 * n - 0.021 * n ** 2) * 255);
    const b = Math.round((0.329 + 0.636 * n - 0.432 * n ** 2) * 255);
    return `rgb(${Math.min(255,r)},${Math.min(255,g)},${Math.min(255,b)})`;
  },
  Plasma: (v) => {
    const n = (v + 1) / 2;
    const r = Math.round((0.05 + 2.0 * n - n ** 2 * 1.0) * 255);
    const g = Math.round((n ** 2 * 0.8) * 255);
    const b = Math.round((0.5 - n * 0.5) * 255);
    return `rgb(${Math.min(255,Math.max(0,r))},${Math.min(255,Math.max(0,g))},${Math.min(255,Math.max(0,b))})`;
  },
};

function normalize(matrix) {
  let min = Infinity;
  let max = -Infinity;
  for (const row of matrix) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;
  return {
    normalized: matrix.map((row) => row.map((v) => ((v - min) / range) * 2 - 1)),
    min,
    max,
  };
}

export default function LFCCHeatmap({ lfccData, lfccShape, colorScheme = 'BlueRed', showValues = false }) {
  const [tooltip, setTooltip] = useState(null);
  const [scheme, setScheme] = useState(colorScheme);

  const { normalized, min, max } = useMemo(() => {
    if (!lfccData || lfccData.length === 0)
      return { normalized: [], min: 0, max: 0 };
    return normalize(lfccData);
  }, [lfccData]);

  const colorFn = COLOR_SCHEMES[scheme] ?? COLOR_SCHEMES.BlueRed;

  const xLabels = useMemo(() => {
    if (!lfccData || lfccData.length === 0) return [];
    const frames = lfccData[0]?.length ?? 0;
    return Array.from({ length: frames }, (_, i) =>
      i % 10 === 0 ? String(i) : ''
    );
  }, [lfccData]);

  const yLabels = useMemo(() => {
    if (!lfccData || lfccData.length === 0) return [];
    return lfccData.map((_, i) => (i % 10 === 0 ? `C${i}` : ''));
  }, [lfccData]);

  if (!lfccData || lfccData.length === 0) {
    return (
      <div className="panel heatmap-panel heatmap-empty">
        <div className="panel-header">
          <h2 className="panel-title">LFCC Feature Map</h2>
        </div>
        <div className="heatmap-placeholder">
          <span className="placeholder-icon">⬛</span>
          <p>Waiting for buffer to fill…</p>
          <p className="placeholder-sub">LFCC data arrives after the first 1.5 s buffer is complete.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel heatmap-panel">
      <div className="panel-header">
        <h2 className="panel-title">LFCC Feature Map</h2>
        <span className="info-chip">
          {lfccShape ? `${lfccShape[0]} × ${lfccShape[1]}` : '–'}
        </span>
        <select
          className="scheme-select"
          value={scheme}
          onChange={(e) => setScheme(e.target.value)}
          id="lfcc-color-scheme"
        >
          {Object.keys(COLOR_SCHEMES).map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <div className="heatmap-legend">
        <span style={{ color: colorFn(-1) }}>● {min.toFixed(2)}</span>
        <span className="legend-mid">Value range</span>
        <span style={{ color: colorFn(1) }}>● {max.toFixed(2)}</span>
      </div>

      <div
        className="heatmap-wrap"
        onMouseLeave={() => setTooltip(null)}
      >
        <HeatMap
          xLabels={xLabels}
          yLabels={yLabels}
          data={normalized}
          squares={false}
          height={8}
          cellStyle={(bg, value, min2, max2, dataRef, x, y) => ({
            background: colorFn(value),
            fontSize: showValues ? '5px' : '0px',
            color: '#fff',
            cursor: 'crosshair',
            padding: '1px',
          })}
          cellRender={(value, x, y) => {
            const raw = lfccData[x]?.[y];
            return (
              <div
                title={raw !== undefined ? `C${x}, F${y}: ${raw.toFixed(4)}` : ''}
                style={{ width: '100%', height: '100%' }}
              />
            );
          }}
        />
      </div>

      {tooltip && (
        <div className="heatmap-tooltip" style={{ top: tooltip.y, left: tooltip.x }}>
          C{tooltip.row}, F{tooltip.col}: {tooltip.value.toFixed(4)}
        </div>
      )}
    </div>
  );
}
