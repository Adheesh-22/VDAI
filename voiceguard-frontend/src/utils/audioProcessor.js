export function downsampleWaveform(float32, targetPoints = 256) {
  if (!float32 || float32.length === 0) return new Array(targetPoints).fill(0);
  const step = Math.max(1, Math.floor(float32.length / targetPoints));
  const out = [];
  for (let i = 0; i < targetPoints; i++) {
    let min = Infinity;
    let max = -Infinity;
    for (let j = 0; j < step; j++) {
      const v = float32[i * step + j] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    out.push({ min: isFinite(min) ? min : 0, max: isFinite(max) ? max : 0 });
  }
  return out;
}

export function rmsVolume(float32) {
  if (!float32 || float32.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < float32.length; i++) sum += float32[i] ** 2;
  return Math.sqrt(sum / float32.length);
}
