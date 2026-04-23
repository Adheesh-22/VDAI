import { useEffect, useRef, useCallback, useState } from 'react';

const RECONNECT_BASE_MS = 3000;
const MAX_RECONNECT_MS = 30000;

export function useWebSocket(url) {
  const wsRef = useRef(null);
  const retryTimerRef = useRef(null);
  const retryCountRef = useRef(0);
  const sentTimeRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const [latencyMs, setLatencyMs] = useState(null);
  const [messageCount, setMessageCount] = useState(0);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState < 2) return;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retryCountRef.current = 0;
      setRetryCount(0);
    };

    ws.onmessage = (event) => {
      const now = performance.now();
      if (sentTimeRef.current !== null) {
        setLatencyMs(Math.round(now - sentTimeRef.current));
        sentTimeRef.current = null;
      }
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        setMessageCount((c) => c + 1);
      } catch {
        /* ignore non-JSON frames */
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      const delay = Math.min(
        RECONNECT_BASE_MS * 2 ** retryCountRef.current,
        MAX_RECONNECT_MS
      );
      retryCountRef.current += 1;
      setRetryCount(retryCountRef.current);
      retryTimerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(retryTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sentTimeRef.current = performance.now();
      wsRef.current.send(data);
      return true;
    }
    return false;
  }, []);

  return { connected, lastMessage, retryCount, latencyMs, messageCount, send };
}
