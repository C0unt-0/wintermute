import { useState, useRef, useCallback, useEffect } from "react";
import { api } from "../api/client.ts";
import ConfidenceBar from "../components/ConfidenceBar.tsx";

interface ScanResult {
  is_malicious: boolean;
  threat_score: number;
  predicted_family: string;
  telemetry: {
    instructions_analyzed: number;
    cfg_nodes: number;
    cfg_edges: number;
  };
}

export default function Scan() {
  const [file, setFile] = useState<File | null>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFile(dropped);
    },
    [handleFile],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) handleFile(selected);
    },
    [handleFile],
  );

  const handleScan = useCallback(async () => {
    if (!file) return;
    setScanning(true);
    setError(null);
    setResult(null);

    try {
      const { job_id } = await api.scan(file);

      pollRef.current = setInterval(async () => {
        try {
          const status = await api.scanStatus(job_id);

          if (status.status === "SUCCESS" || status.status === "COMPLETED") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setResult(status.result as unknown as ScanResult);
            setScanning(false);
          } else if (status.status === "FAILED") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setError("Scan failed. Please try again.");
            setScanning(false);
          }
        } catch {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setError("Failed to poll scan status.");
          setScanning(false);
        }
      }, 1000);
    } catch {
      setError("Failed to start scan.");
      setScanning(false);
    }
  }, [file]);

  const isMalicious = result?.is_malicious ?? false;
  const verdict = isMalicious ? "MALICIOUS" : "SAFE";
  const confidence = result?.threat_score ?? 0;
  const verdictColor = isMalicious ? "var(--threat)" : "var(--safe)";

  return (
    <div className="p-6 h-full">
      <h2
        className="text-sm uppercase tracking-widest mb-6"
        style={{ fontFamily: "var(--font-heading)", color: "var(--data)" }}
      >
        Scan
      </h2>

      <div className="grid grid-cols-3 gap-4" style={{ height: "calc(100vh - 140px)" }}>
        {/* Left column: Disassembly + Drop Zone */}
        <div className="col-span-2 flex flex-col gap-4 min-h-0">
          {/* Disassembly View */}
          <div
            className="flex-1 rounded-lg p-4 overflow-auto min-h-0"
            style={{
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border)",
              fontFamily: "var(--font-code)",
            }}
          >
            {result ? (
              <div className="text-xs leading-relaxed" style={{ color: "var(--text-primary)" }}>
                <div className="mb-3" style={{ color: "var(--text-muted)" }}>
                  ; ── Analysis Results ──────────────────────────
                </div>
                <div className="mb-4">
                  <span style={{ color: "var(--text-muted)" }}>; Verdict: </span>
                  <span style={{ color: verdictColor, fontWeight: 700 }}>
                    {verdict}
                  </span>
                </div>
                <div className="mb-3" style={{ color: "var(--text-muted)" }}>
                  ; ── Telemetry ─────────────────────────────────
                </div>
                <div className="space-y-1">
                  <div>
                    <span style={{ color: "var(--purple)" }}>instructions_analyzed</span>
                    <span style={{ color: "var(--text-muted)" }}>{" : "}</span>
                    <span style={{ color: "var(--data)" }}>
                      {result.telemetry.instructions_analyzed.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span style={{ color: "var(--purple)" }}>cfg_nodes</span>
                    <span style={{ color: "var(--text-muted)" }}>{" : "}</span>
                    <span style={{ color: "var(--data)" }}>
                      {result.telemetry.cfg_nodes.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span style={{ color: "var(--purple)" }}>cfg_edges</span>
                    <span style={{ color: "var(--text-muted)" }}>{" : "}</span>
                    <span style={{ color: "var(--data)" }}>
                      {result.telemetry.cfg_edges.toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="mt-4 mb-3" style={{ color: "var(--text-muted)" }}>
                  ; ── Scores ──────────────────────────────────
                </div>
                <div className="space-y-1">
                  <div>
                    <span style={{ color: "var(--purple)" }}>confidence</span>
                    <span style={{ color: "var(--text-muted)" }}>{" : "}</span>
                    <span style={{ color: verdictColor }}>
                      {(confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span style={{ color: "var(--purple)" }}>threat_score</span>
                    <span style={{ color: "var(--text-muted)" }}>{" : "}</span>
                    <span style={{ color: isMalicious ? "var(--threat)" : "var(--safe)" }}>
                      {(result.threat_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span style={{ color: "var(--purple)" }}>predicted_family</span>
                    <span style={{ color: "var(--text-muted)" }}>{" : "}</span>
                    <span style={{ color: "var(--warn)" }}>
                      {result.predicted_family || "N/A"}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div
                className="h-full flex items-center justify-center text-sm"
                style={{ color: "var(--text-muted)", fontFamily: "var(--font-code)" }}
              >
                No scan results yet
              </div>
            )}
          </div>

          {/* Drop Zone */}
          <div
            className="rounded-lg p-8 flex flex-col items-center justify-center cursor-pointer transition-all shrink-0"
            style={{
              backgroundColor: "var(--bg-surface)",
              border: `2px dashed ${dragOver ? "var(--data)" : "var(--border)"}`,
              minHeight: "160px",
              ...(dragOver ? { boxShadow: "0 0 20px rgba(0, 212, 255, 0.1)" } : {}),
            }}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleInputChange}
            />

            {/* File icon */}
            <svg
              width="40"
              height="40"
              viewBox="0 0 24 24"
              fill="none"
              stroke={dragOver ? "var(--data)" : "var(--text-muted)"}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mb-3 transition-colors"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <line x1="9" y1="15" x2="12" y2="12" />
              <line x1="15" y1="15" x2="12" y2="12" />
            </svg>

            {file ? (
              <div className="text-center">
                <p
                  className="text-sm mb-1"
                  style={{ fontFamily: "var(--font-code)", color: "var(--data)" }}
                >
                  {file.name}
                </p>
                <p
                  className="text-xs mb-4"
                  style={{ color: "var(--text-muted)" }}
                >
                  {(file.size / 1024).toFixed(1)} KB
                </p>
                <button
                  className="px-6 py-2 rounded text-xs font-bold uppercase tracking-widest transition-all"
                  style={{
                    fontFamily: "var(--font-heading)",
                    backgroundColor: scanning ? "var(--bg-elevated)" : "var(--data)",
                    color: scanning ? "var(--text-muted)" : "var(--bg-primary)",
                    cursor: scanning ? "not-allowed" : "pointer",
                    boxShadow: scanning ? "none" : "0 0 16px rgba(0, 212, 255, 0.3)",
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!scanning) handleScan();
                  }}
                  disabled={scanning}
                >
                  {scanning ? "Scanning..." : "Scan"}
                </button>
              </div>
            ) : (
              <p
                className="text-sm"
                style={{ color: "var(--text-muted)", fontFamily: "var(--font-code)" }}
              >
                Drop a binary here or click to upload
              </p>
            )}
          </div>
        </div>

        {/* Right column: Verdict Panel */}
        <div
          className="col-span-1 rounded-lg p-6 flex flex-col overflow-auto"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          {result ? (
            <>
              {/* Verdict */}
              <div className="text-center mb-8">
                <div
                  className="text-3xl font-bold mb-2"
                  style={{
                    fontFamily: "var(--font-heading)",
                    color: verdictColor,
                    textShadow: `0 0 20px ${verdictColor}40`,
                  }}
                >
                  {verdict}
                </div>
                <div
                  className="text-xs uppercase tracking-widest"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Verdict
                </div>
              </div>

              {/* Confidence Bars */}
              <div className="mb-6">
                <p
                  className="text-[10px] uppercase tracking-widest mb-3"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Confidence
                </p>
                <div className="space-y-3">
                  <ConfidenceBar
                    value={isMalicious ? confidence : 1 - confidence}
                    label="Malicious"
                    color="var(--threat)"
                  />
                  <ConfidenceBar
                    value={isMalicious ? 1 - confidence : confidence}
                    label="Safe"
                    color="var(--safe)"
                  />
                </div>
              </div>

              {/* Threat Score */}
              <div className="mb-6">
                <p
                  className="text-[10px] uppercase tracking-widest mb-3"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Threat Score
                </p>
                <ConfidenceBar
                  value={result.threat_score}
                  color={result.threat_score > 0.5 ? "var(--threat)" : "var(--safe)"}
                />
              </div>

              {/* Predicted Family */}
              {result.predicted_family && (
                <div className="mb-6">
                  <p
                    className="text-[10px] uppercase tracking-widest mb-2"
                    style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                  >
                    Predicted Family
                  </p>
                  <p
                    className="text-sm font-bold"
                    style={{ fontFamily: "var(--font-code)", color: "var(--warn)" }}
                  >
                    {result.predicted_family}
                  </p>
                </div>
              )}

              {/* Metadata */}
              <div className="mt-auto pt-4" style={{ borderTop: "1px solid var(--border)" }}>
                <p
                  className="text-[10px] uppercase tracking-widest mb-3"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Metadata
                </p>
                <div className="space-y-2">
                  <MetadataRow
                    label="Instructions"
                    value={result.telemetry.instructions_analyzed.toLocaleString()}
                  />
                  <MetadataRow
                    label="CFG Nodes"
                    value={result.telemetry.cfg_nodes.toLocaleString()}
                  />
                  <MetadataRow
                    label="CFG Edges"
                    value={result.telemetry.cfg_edges.toLocaleString()}
                  />
                </div>
              </div>
            </>
          ) : (
            <div
              className="h-full flex flex-col items-center justify-center text-center"
              style={{ color: "var(--text-muted)" }}
            >
              <svg
                width="48"
                height="48"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mb-4 opacity-30"
              >
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <p
                className="text-sm mb-1"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                No Results
              </p>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Upload and scan a binary to see results
              </p>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div
              className="mt-4 rounded p-3 text-xs"
              style={{
                backgroundColor: "rgba(255, 59, 92, 0.1)",
                border: "1px solid var(--threat)",
                color: "var(--threat)",
                fontFamily: "var(--font-code)",
              }}
            >
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span
        className="text-xs"
        style={{ fontFamily: "var(--font-code)", color: "var(--text-muted)" }}
      >
        {label}
      </span>
      <span
        className="text-xs font-bold"
        style={{ fontFamily: "var(--font-code)", color: "var(--data)" }}
      >
        {value}
      </span>
    </div>
  );
}
