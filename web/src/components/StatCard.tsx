interface StatCardProps {
  label: string;
  value: string;
  subtitle?: string;
  color?: string;
}

export default function StatCard({ label, value, subtitle, color = "var(--data)" }: StatCardProps) {
  return (
    <div
      className="rounded-lg p-4"
      style={{
        backgroundColor: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        boxShadow: `0 0 12px ${color}15`,
      }}
    >
      <p
        className="text-[10px] uppercase tracking-widest mb-2"
        style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
      >
        {label}
      </p>
      <p
        className="text-2xl font-bold leading-tight"
        style={{ fontFamily: "var(--font-code)", color }}
      >
        {value}
      </p>
      {subtitle && (
        <p
          className="text-xs mt-1"
          style={{ color: "var(--text-muted)" }}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}
