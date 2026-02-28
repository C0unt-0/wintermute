import { useEffect, useRef, useState, useCallback } from "react";
import { createWebSocket, type WebSocketHandle } from "../api/ws";

export interface WSEvent {
  type: string;
  [key: string]: unknown;
}

export function useWebSocket() {
  const [events, setEvents] = useState<WSEvent[]>([]);
  const handleRef = useRef<WebSocketHandle | null>(null);
  const listenersRef = useRef<Map<string, Set<(data: WSEvent) => void>>>(
    new Map(),
  );

  useEffect(() => {
    handleRef.current = createWebSocket((data) => {
      const event = data as WSEvent;
      setEvents((prev) => [...prev.slice(-200), event]);

      // Notify type-specific listeners
      const listeners = listenersRef.current.get(event.type);
      if (listeners) {
        listeners.forEach((fn) => fn(event));
      }
    });

    return () => {
      handleRef.current?.destroy();
    };
  }, []);

  const subscribe = useCallback(
    (type: string, handler: (data: WSEvent) => void) => {
      if (!listenersRef.current.has(type)) {
        listenersRef.current.set(type, new Set());
      }
      listenersRef.current.get(type)!.add(handler);

      // Return unsubscribe function
      return () => {
        listenersRef.current.get(type)?.delete(handler);
      };
    },
    [],
  );

  return { events, subscribe };
}
