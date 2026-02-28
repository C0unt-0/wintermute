export type WSEventHandler = (data: Record<string, unknown>) => void;

export interface WebSocketHandle {
  ws: WebSocket;
  destroy: () => void;
}

export function createWebSocket(onMessage: WSEventHandler): WebSocketHandle {
  let shouldReconnect = true;
  let backoff = 2000;
  const MAX_BACKOFF = 30000;

  function connect(): WebSocket {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${protocol}//${window.location.host}/api/v1/ws`,
    );

    ws.onopen = () => {
      // Reset backoff on successful connection
      backoff = 2000;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as Record<string, unknown>;
        onMessage(data);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (shouldReconnect) {
        setTimeout(() => {
          if (shouldReconnect) {
            socket = connect();
          }
        }, backoff);
        backoff = Math.min(backoff * 2, MAX_BACKOFF);
      }
    };

    return ws;
  }

  let socket = connect();

  return {
    get ws() {
      return socket;
    },
    destroy() {
      shouldReconnect = false;
      socket.close();
    },
  };
}
