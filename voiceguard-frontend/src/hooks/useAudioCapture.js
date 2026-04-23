import { useRef, useState, useCallback, useEffect } from 'react';

const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 4096;
const WAVEFORM_SIZE = 512;

export function useAudioCapture(sendFn) {
  const audioCtxRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);
  const pausedRef = useRef(false);
  const chunkCountRef = useRef(0);

  const [recording, setRecording] = useState(false);
  const [paused, setPaused] = useState(false);
  const [volume, setVolume] = useState(0);
  const [waveform, setWaveform] = useState(new Float32Array(WAVEFORM_SIZE));
  const [chunkCount, setChunkCount] = useState(0);
  const [error, setError] = useState(null);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = ctx.createScriptProcessor(CHUNK_SIZE, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0);

        let rms = 0;
        for (let i = 0; i < float32.length; i++) rms += float32[i] ** 2;
        rms = Math.sqrt(rms / float32.length);
        setVolume(Math.min(1, rms * 5));

        const step = Math.floor(float32.length / WAVEFORM_SIZE);
        const snap = new Float32Array(WAVEFORM_SIZE);
        for (let i = 0; i < WAVEFORM_SIZE; i++) snap[i] = float32[i * step] || 0;
        setWaveform(snap);

        if (!pausedRef.current) {
          sendFn(float32.buffer.slice(0));
          chunkCountRef.current += 1;
          setChunkCount(chunkCountRef.current);
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);
      setRecording(true);
      pausedRef.current = false;
      setPaused(false);
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setError('microphone_denied');
      } else {
        setError(err.message || 'audio_init_failed');
      }
    }
  }, [sendFn]);

  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    audioCtxRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    processorRef.current = null;
    sourceRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
    chunkCountRef.current = 0;
    setRecording(false);
    setPaused(false);
    setVolume(0);
    setWaveform(new Float32Array(WAVEFORM_SIZE));
    setChunkCount(0);
  }, []);

  const togglePause = useCallback(() => {
    pausedRef.current = !pausedRef.current;
    setPaused(pausedRef.current);
  }, []);

  useEffect(() => {
    return () => {
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      audioCtxRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    recording,
    paused,
    volume,
    waveform,
    chunkCount,
    error,
    startRecording,
    stopRecording,
    togglePause,
  };
}
