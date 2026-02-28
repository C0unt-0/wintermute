import { useState, useEffect, useMemo, useCallback } from "react";
import ConfigPanel from "../components/ConfigPanel.tsx";
import type { FieldDef } from "../components/ConfigPanel.tsx";
import ActivityLog from "../components/ActivityLog.tsx";
import type { LogEntry } from "../components/ActivityLog.tsx";
import { useJob } from "../hooks/useJob.ts";
import { useWebSocket } from "../hooks/useWebSocket.ts";
import type { WSEvent } from "../hooks/useWebSocket.ts";
import { api } from "../api/client.ts";
import type { PipelineConfig } from "../api/client.ts";

type Operation = "build" | "synthetic" | "pretrain";

const operationLabels: Record<Operation, string> = {
  build: "Build Dataset",
  synthetic: "Synthetic Data",
  pretrain: "MalBERT Pretrain",
};

const buildFields: FieldDef[] = [
  { name: "data_dir", label: "Data Directory", type: "text", default: "data" },
  { name: "max_seq_length", label: "Max Seq Length", type: "number", default: 2048 },
];

const syntheticFields: FieldDef[] = [
  { name: "n_samples", label: "Samples", type: "number", default: 500 },
  { name: "output_dir", label: "Output Dir", type: "text", default: "data/processed" },
  { name: "seed", label: "Seed", type: "number", default: 42 },
];

const pretrainFields: FieldDef[] = [
  { name: "epochs", label: "Epochs", type: "number", default: 50 },
  { name: "learning_rate", label: "Learning Rate", type: "number", default: 0.0003 },
  { name: "batch_size", label: "Batch Size", type: "number", default: 8 },
  { name: "mask_prob", label: "Mask Probability", type: "number", default: 0.15 },
];

const fieldsByOperation: Record<Operation, FieldDef[]> = {
  build: buildFields,
  synthetic: syntheticFields,
  pretrain: pretrainFields,
};

export default function Pipeline() {
  const [operation, setOperation] = useState<Operation>("synthetic");
  const [progress, setProgress] = useState(0);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [configOpen, setConfigOpen] = useState(true);

  const { subscribe } = useWebSocket();

  // Capture current operation in a ref-like stable callback for the start function
  const currentOperation = operation;

  const startFn = useCallback(
    (config: PipelineConfig) => api.startPipeline(currentOperation, config),
    [currentOperation],
  );
  const pollFn = useCallback((id: string) => api.pipelineStatus(id), []);
  const cancelFn = useCallback((id: string) => api.cancelPipeline(id), []);

  const { isRunning, error, startJob, cancelJob } = useJob({
    start: startFn,
    poll: pollFn,
    cancel: cancelFn,
  });

  // Subscribe to pipeline_progress WebSocket events
  useEffect(() => {
    const unsubscribe = subscribe("pipeline_progress", (e: WSEvent) => {
      setProgress(e.progress as number);
      const timestamp = new Date().toLocaleTimeString();
      setLogEntries((prev) => [
        ...prev,
        { text: e.message as string, level: "info", timestamp },
      ]);
    });
    return unsubscribe;
  }, [subscribe]);

  const activeFields = useMemo(
    () => fieldsByOperation[operation],
    [operation],
  );

  const handleStart = (values: Record<string, string | number | boolean>) => {
    setProgress(0);
    setLogEntries([]);
    const config: PipelineConfig = {};
    for (const [key, val] of Object.entries(values)) {
      (config as Record<string, string | number | boolean>)[key] = val;
    }
    void startJob(config);
  };

  return (
    <div className="flex h-full">
      {/* Main content area */}
      <div
        className="flex-1 overflow-auto p-8"
        style={{
          marginRight: configOpen ? "20rem" : 0,
          transition: "margin-right 300ms ease-in-out",
        }}
      >
        {/* Page heading */}
        <h2
          className="text-xl mb-6"
          style={{ fontFamily: "var(--font-heading)", color: "var(--text-primary)" }}
        >
          Pipeline
        </h2>

        {/* Error banner */}
        {error && (
          <div
            className="mb-4 px-4 py-2 rounded text-sm"
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

        {/* Operation selector + status */}
        <div
          className="rounded p-5 mb-6"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          <div className="flex items-center gap-6 flex-wrap">
            {/* Operation dropdown */}
            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Operation
              </span>
              <select
                value={operation}
                disabled={isRunning}
                onChange={(e) => setOperation(e.target.value as Operation)}
                className="px-3 py-2 text-xs rounded border outline-none disabled:opacity-50"
                style={{
                  fontFamily: "var(--font-code)",
                  backgroundColor: "var(--bg-elevated)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                  minWidth: "180px",
                }}
              >
                {(Object.keys(operationLabels) as Operation[]).map((op) => (
                  <option key={op} value={op}>
                    {operationLabels[op]}
                  </option>
                ))}
              </select>
            </div>

            {/* Status */}
            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Status
              </span>
              <span
                className="text-sm font-bold"
                style={{
                  fontFamily: "var(--font-code)",
                  color: isRunning ? "var(--safe)" : "var(--text-muted)",
                }}
              >
                {isRunning ? "RUNNING" : progress >= 100 ? "COMPLETED" : "IDLE"}
              </span>
            </div>

            {/* Progress percentage */}
            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Progress
              </span>
              <span
                className="text-sm font-bold"
                style={{ fontFamily: "var(--font-code)", color: "var(--data)" }}
              >
                {progress}%
              </span>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div
          className="rounded p-4 mb-6"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          <span
            className="text-[10px] uppercase tracking-widest block mb-2"
            style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
          >
            Pipeline Progress
          </span>
          <div
            className="w-full h-5 rounded overflow-hidden"
            style={{ backgroundColor: "var(--bg-elevated)" }}
          >
            <div
              className="h-full rounded"
              style={{
                width: `${Math.min(progress, 100)}%`,
                backgroundColor: "var(--data)",
                transition: "width 400ms ease-in-out",
              }}
            />
          </div>
          <span
            className="text-xs mt-1 block text-right"
            style={{ fontFamily: "var(--font-code)", color: "var(--text-muted)" }}
          >
            {progress}%
          </span>
        </div>

        {/* Log output */}
        <div
          className="rounded p-4 mb-6"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          <span
            className="text-[10px] uppercase tracking-widest block mb-2"
            style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
          >
            Log Output
          </span>
          <ActivityLog entries={logEntries} maxHeight="360px" />
        </div>

        {/* Empty state */}
        {!isRunning && logEntries.length === 0 && progress === 0 && (
          <div
            className="flex flex-col items-center justify-center py-20 rounded"
            style={{
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border)",
            }}
          >
            <span
              className="text-sm mb-2"
              style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
            >
              NO PIPELINE ACTIVITY
            </span>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              Select an operation, configure parameters, and press START
            </span>
          </div>
        )}
      </div>

      {/* Config panel */}
      <ConfigPanel
        title={`${operationLabels[operation]} Config`}
        fields={activeFields}
        onStart={handleStart}
        onCancel={() => void cancelJob()}
        isRunning={isRunning}
        isOpen={configOpen}
        onToggle={() => setConfigOpen((prev) => !prev)}
      />
    </div>
  );
}
