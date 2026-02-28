export type WSEventHandler = (data: Record<string, unknown>) => void;

export function createWebSocket(onMessage: WSEventHandler): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(
    `${protocol}//${window.location.host}/api/v1/ws`,
  );

  ws.onmessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data as string) as Record<string, unknown>;
      onMessage(data);
    } catch {
      // Ignore malformed messages
    }
  };

  ws.onclose = () => {
    // Auto-reconnect after 2 seconds
    setTimeout(() => createWebSocket(onMessage), 2000);
  };

  return ws;
}
