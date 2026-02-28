import { useState, useEffect, useCallback } from "react";
import DiffView from "../components/DiffView.tsx";
import { useWebSocket } from "../hooks/useWebSocket.ts";
import { api } from "../api/client.ts";
import type { VaultSample, VaultSampleDetail } from "../api/client.ts";

export default function Vault() {
  const [samples, setSamples] = useState<VaultSample[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<VaultSampleDetail | null>(null);
  const [loading, setLoading] = useState(true);

  const { subscribe } = useWebSocket();

  // Fetch sample list on mount
  useEffect(() => {
    api
      .vaultSamples()
      .then((data) => setSamples(data))
      .catch(() => {
        /* silently handle — empty state will show */
      })
      .finally(() => setLoading(false));
  }, []);

  // Subscribe to real-time vault_sample_added events
  useEffect(() => {
    const unsubscribe = subscribe("vault_sample_added", (e) => {
      const sample = e as unknown as VaultSample;
      setSamples((prev) => [...prev, sample]);
    });
    return unsubscribe;
  }, [subscribe]);

  // Fetch detail when a sample is selected
  const handleSelect = useCallback(
    (id: string) => {
      setSelectedId(id);
      setDetail(null);
      api
        .vaultSample(id)
        .then((data) => setDetail(data))
        .catch(() => setDetail(null));
    },
    [],
  );

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center h-full">
        <p
          className="text-sm"
          style={{ color: "var(--text-muted)", fontFamily: "var(--font-code)" }}
        >
          Loading vault...
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Page heading */}
      <h2
        className="text-xl"
        style={{ fontFamily: "var(--font-heading)", color: "var(--text-primary)" }}
      >
        Vault
      </h2>

      {/* Two-column layout */}
      <div className="grid grid-cols-2 gap-6" style={{ minHeight: "calc(100vh - 180px)" }}>
        {/* Left panel — Sample table */}
        <div
          className="rounded-lg overflow-hidden flex flex-col"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          <div
            className="px-4 py-3"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <span
              className="text-[10px] uppercase tracking-widest"
              style={{
                fontFamily: "var(--font-heading)",
                color: "var(--text-muted)",
              }}
            >
              Vault Samples
            </span>
          </div>

          {samples.length === 0 ? (
            <div className="flex-1 flex items-center justify-center p-8">
              <p
                className="text-sm"
                style={{
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-code)",
                }}
              >
                No adversarial samples in vault
              </p>
            </div>
          ) : (
            <div className="flex-1 overflow-auto">
              <table
                className="w-full"
                style={{ fontFamily: "var(--font-code)" }}
              >
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    {["ID", "Family", "Conf", "Mut", "Cyc"].map((header) => (
                      <th
                        key={header}
                        className="px-4 py-2 text-left text-[10px] uppercase tracking-widest"
                        style={{
                          fontFamily: "var(--font-heading)",
                          color: "var(--text-muted)",
                          backgroundColor: "var(--bg-elevated)",
                          position: "sticky",
                          top: 0,
                        }}
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {samples.map((sample) => {
                    const isSelected = sample.id === selectedId;
                    return (
                      <tr
                        key={sample.id}
                        onClick={() => handleSelect(sample.id)}
                        style={{
                          borderBottom: "1px solid var(--border)",
                          backgroundColor: isSelected
                            ? "var(--bg-elevated)"
                            : "transparent",
                          cursor: "pointer",
                          transition: "background-color 150ms ease",
                        }}
                        onMouseEnter={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.backgroundColor =
                              "rgba(255, 255, 255, 0.02)";
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.backgroundColor = "transparent";
                          }
                        }}
                      >
                        <td
                          className="px-4 py-2 text-xs"
                          style={{ color: "var(--data)" }}
                        >
                          {sample.id.slice(0, 8)}
                        </td>
                        <td
                          className="px-4 py-2 text-xs"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {sample.family}
                        </td>
                        <td
                          className="px-4 py-2 text-xs"
                          style={{ color: "var(--warn)" }}
                        >
                          {sample.confidence.toFixed(2)}
                        </td>
                        <td
                          className="px-4 py-2 text-xs"
                          style={{ color: "var(--purple)" }}
                        >
                          {sample.mutations}
                        </td>
                        <td
                          className="px-4 py-2 text-xs"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {sample.cycle}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right panel — Sample detail */}
        <div
          className="rounded-lg overflow-hidden flex flex-col"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          <div
            className="px-4 py-3"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <span
              className="text-[10px] uppercase tracking-widest"
              style={{
                fontFamily: "var(--font-heading)",
                color: "var(--text-muted)",
              }}
            >
              Sample Detail
            </span>
          </div>

          {!selectedId ? (
            <div className="flex-1 flex items-center justify-center p-8">
              <p
                className="text-sm"
                style={{
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-code)",
                }}
              >
                Select a sample to view details
              </p>
            </div>
          ) : !detail ? (
            <div className="flex-1 flex items-center justify-center p-8">
              <p
                className="text-sm"
                style={{
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-code)",
                }}
              >
                Loading...
              </p>
            </div>
          ) : (
            <div className="flex-1 overflow-auto p-4 space-y-5">
              {/* Metadata section */}
              <div className="space-y-3">
                {[
                  { label: "ID", value: detail.id, color: "var(--data)" },
                  { label: "Family", value: detail.family, color: "var(--text-primary)" },
                  {
                    label: "Confidence",
                    value: `${(detail.confidence * 100).toFixed(0)}%`,
                    color: "var(--warn)",
                  },
                  {
                    label: "Mutations",
                    value: String(detail.mutations),
                    color: "var(--purple)",
                  },
                  {
                    label: "Cycle",
                    value: String(detail.cycle),
                    color: "var(--text-muted)",
                  },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex items-baseline gap-3">
                    <span
                      className="text-[10px] uppercase tracking-widest w-24 shrink-0"
                      style={{
                        fontFamily: "var(--font-heading)",
                        color: "var(--text-muted)",
                      }}
                    >
                      {label}
                    </span>
                    <span
                      className="text-sm"
                      style={{
                        fontFamily: "var(--font-code)",
                        color,
                      }}
                    >
                      {value}
                    </span>
                  </div>
                ))}
              </div>

              {/* Mutation Diff section */}
              <div>
                <span
                  className="text-[10px] uppercase tracking-widest block mb-2"
                  style={{
                    fontFamily: "var(--font-heading)",
                    color: "var(--text-muted)",
                  }}
                >
                  Mutation Diff
                </span>
                <DiffView diff={detail.diff} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
