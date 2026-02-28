import { useState, useEffect } from "react";

interface FieldDef {
  name: string;
  label: string;
  type: "number" | "text" | "select" | "toggle";
  default: string | number | boolean;
  options?: string[];
}

interface ConfigPanelProps {
  title: string;
  fields: FieldDef[];
  onStart: (values: Record<string, string | number | boolean>) => void;
  onCancel?: () => void;
  isRunning?: boolean;
  isOpen: boolean;
  onToggle: () => void;
}

export type { FieldDef };

export default function ConfigPanel({
  title,
  fields,
  onStart,
  onCancel,
  isRunning = false,
  isOpen,
  onToggle,
}: ConfigPanelProps) {
  const [values, setValues] = useState<Record<string, string | number | boolean>>({});

  useEffect(() => {
    const defaults: Record<string, string | number | boolean> = {};
    for (const field of fields) {
      defaults[field.name] = field.default;
    }
    setValues(defaults);
  }, [fields]);

  const handleChange = (name: string, raw: string | boolean, type: FieldDef["type"]) => {
    setValues((prev) => ({
      ...prev,
      [name]: type === "number" ? Number(raw) : raw,
    }));
  };

  return (
    <>
      {/* Toggle button (always visible) */}
      <button
        onClick={onToggle}
        className="fixed top-14 right-0 z-40 px-2 py-4 text-[10px] tracking-widest uppercase rounded-l-md border border-r-0 cursor-pointer"
        style={{
          fontFamily: "var(--font-heading)",
          backgroundColor: "var(--bg-elevated)",
          borderColor: "var(--border)",
          color: "var(--text-muted)",
          writingMode: "vertical-rl",
        }}
      >
        {isOpen ? "CLOSE" : "CONFIG"}
      </button>

      {/* Slide-in panel */}
      <div
        className="fixed top-12 right-0 z-30 h-[calc(100vh-3rem)] w-80 flex flex-col border-l transition-transform duration-300 ease-in-out"
        style={{
          backgroundColor: "var(--bg-surface)",
          borderColor: "var(--border)",
          transform: isOpen ? "translateX(0)" : "translateX(100%)",
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-3 border-b"
          style={{ borderColor: "var(--border)" }}
        >
          <h3
            className="text-sm tracking-wider uppercase"
            style={{ fontFamily: "var(--font-heading)", color: "var(--text-primary)" }}
          >
            {title}
          </h3>
        </div>

        {/* Form fields */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {fields.map((field) => (
            <div key={field.name}>
              <label
                className="block text-[10px] uppercase tracking-widest mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                {field.label}
              </label>

              {field.type === "toggle" ? (
                <button
                  disabled={isRunning}
                  onClick={() => handleChange(field.name, !values[field.name], "toggle")}
                  className="relative w-10 h-5 rounded-full transition-colors duration-200 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: values[field.name] ? "var(--safe)" : "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <span
                    className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-transform duration-200"
                    style={{
                      backgroundColor: "var(--text-primary)",
                      transform: values[field.name] ? "translateX(20px)" : "translateX(0)",
                    }}
                  />
                </button>
              ) : field.type === "select" ? (
                <select
                  disabled={isRunning}
                  value={String(values[field.name] ?? field.default)}
                  onChange={(e) => handleChange(field.name, e.target.value, "select")}
                  className="w-full px-3 py-2 text-xs rounded border outline-none disabled:opacity-50"
                  style={{
                    fontFamily: "var(--font-code)",
                    backgroundColor: "var(--bg-elevated)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                >
                  {field.options?.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={field.type}
                  disabled={isRunning}
                  value={String(values[field.name] ?? field.default)}
                  onChange={(e) => handleChange(field.name, e.target.value, field.type)}
                  className="w-full px-3 py-2 text-xs rounded border outline-none disabled:opacity-50"
                  style={{
                    fontFamily: "var(--font-code)",
                    backgroundColor: "var(--bg-elevated)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
              )}
            </div>
          ))}
        </div>

        {/* Action button */}
        <div className="p-4 border-t" style={{ borderColor: "var(--border)" }}>
          {isRunning ? (
            <button
              onClick={onCancel}
              className="w-full py-2 text-xs uppercase tracking-widest rounded font-bold cursor-pointer transition-opacity hover:opacity-80"
              style={{
                fontFamily: "var(--font-heading)",
                backgroundColor: "var(--threat)",
                color: "var(--bg-primary)",
              }}
            >
              CANCEL
            </button>
          ) : (
            <button
              onClick={() => onStart(values)}
              className="w-full py-2 text-xs uppercase tracking-widest rounded font-bold cursor-pointer transition-opacity hover:opacity-80"
              style={{
                fontFamily: "var(--font-heading)",
                backgroundColor: "var(--safe)",
                color: "var(--bg-primary)",
              }}
            >
              START
            </button>
          )}
        </div>
      </div>
    </>
  );
}
