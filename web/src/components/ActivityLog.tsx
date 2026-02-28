import { useEffect, useRef } from "react";

interface LogEntry {
  text: string;
  level: "info" | "ok" | "warn" | "error";
  timestamp: string;
}

interface ActivityLogProps {
  entries: LogEntry[];
  maxHeight?: string;
}

export type { LogEntry };

const levelColor: Record<LogEntry["level"], string> = {
  info: "var(--text-muted)",
  ok: "var(--safe)",
  warn: "var(--warn)",
  error: "var(--threat)",
};

export default function ActivityLog({ entries, maxHeight = "320px" }: ActivityLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries.length]);

  return (
    <div
      className="overflow-y-auto rounded-lg p-3"
      style={{
        maxHeight,
        backgroundColor: "var(--bg-elevated)",
        border: "1px solid var(--border)",
      }}
    >
      {entries.length === 0 && (
        <p className="text-xs" style={{ color: "var(--text-muted)", fontFamily: "var(--font-code)" }}>
          No events yet.
        </p>
      )}
      {entries.map((entry, i) => (
        <div
          key={i}
          className="text-xs leading-5"
          style={{ fontFamily: "var(--font-code)", color: levelColor[entry.level] }}
        >
          <span style={{ color: "var(--text-muted)" }}>[{entry.timestamp}]</span>{" "}
          {entry.text}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
