interface ConfidenceBarProps {
  value: number;
  label?: string;
  color?: string;
}

export default function ConfidenceBar({ value, label, color = "var(--data)" }: ConfidenceBarProps) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);

  return (
    <div className="w-full">
      {label && (
        <div className="flex items-center justify-between mb-1">
          <span
            className="text-xs"
            style={{ fontFamily: "var(--font-code)", color: "var(--text-muted)" }}
          >
            {label}
          </span>
          <span
            className="text-xs font-bold"
            style={{ fontFamily: "var(--font-code)", color }}
          >
            {pct}%
          </span>
        </div>
      )}
      <div
        className="w-full h-1.5 rounded-full overflow-hidden"
        style={{ backgroundColor: "var(--bg-primary)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
      {!label && (
        <p
          className="text-xs text-right mt-1 font-bold"
          style={{ fontFamily: "var(--font-code)", color }}
        >
          {pct}%
        </p>
      )}
    </div>
  );
}
