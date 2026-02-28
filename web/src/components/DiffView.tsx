interface DiffViewProps {
  diff: string;
}

export default function DiffView({ diff }: DiffViewProps) {
  const lines = diff.split("\n");

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        backgroundColor: "var(--bg-elevated)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <tbody>
            {lines.map((line, i) => {
              let color = "var(--text-muted)";
              let bg = "transparent";

              if (line.startsWith("+")) {
                color = "var(--safe)";
                bg = "rgba(0, 232, 143, 0.06)";
              } else if (line.startsWith("-")) {
                color = "var(--threat)";
                bg = "rgba(255, 59, 92, 0.06)";
              }

              return (
                <tr key={i} style={{ backgroundColor: bg }}>
                  <td
                    className="px-3 py-0 text-right select-none w-10"
                    style={{
                      fontFamily: "var(--font-code)",
                      fontSize: "11px",
                      lineHeight: "20px",
                      color: "var(--text-muted)",
                      opacity: 0.4,
                      borderRight: "1px solid var(--border)",
                    }}
                  >
                    {i + 1}
                  </td>
                  <td
                    className="px-3 py-0 whitespace-pre"
                    style={{
                      fontFamily: "var(--font-code)",
                      fontSize: "11px",
                      lineHeight: "20px",
                      color,
                    }}
                  >
                    {line}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
