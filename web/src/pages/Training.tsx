import { useState, useEffect, useMemo } from "react";
import ConfigPanel from "../components/ConfigPanel.tsx";
import type { FieldDef } from "../components/ConfigPanel.tsx";
import SparklineChart from "../components/SparklineChart.tsx";
import { useJob } from "../hooks/useJob.ts";
import { useWebSocket } from "../hooks/useWebSocket.ts";
import type { WSEvent } from "../hooks/useWebSocket.ts";
import { api } from "../api/client.ts";
import type { TrainingConfig } from "../api/client.ts";

interface EpochData {
  epoch: number;
  phase: string;
  loss: number;
  train_acc: number;
  val_acc: number;
  f1: number;
  elapsed: number;
}

const trainingFields: FieldDef[] = [
  { name: "epochs_phase_a", label: "Epochs Phase A", type: "number", default: 5 },
  { name: "epochs_phase_b", label: "Epochs Phase B", type: "number", default: 20 },
  { name: "learning_rate", label: "Learning Rate", type: "number", default: 0.0003 },
  { name: "batch_size", label: "Batch Size", type: "number", default: 8 },
  {
    name: "max_seq_length",
    label: "Max Seq Length",
    type: "select",
    default: "2048",
    options: ["512", "1024", "2048"],
  },
  {
    name: "num_classes",
    label: "Classes",
    type: "select",
    default: "2",
    options: ["2", "9"],
  },
  { name: "mlflow", label: "MLflow Tracking", type: "toggle", default: false },
  { name: "experiment_name", label: "Experiment", type: "text", default: "default" },
];

export default function Training() {
  const [epochs, setEpochs] = useState<EpochData[]>([]);
  const [losses, setLosses] = useState<number[]>([]);
  const [accuracies, setAccuracies] = useState<number[]>([]);
  const [configOpen, setConfigOpen] = useState(true);

  const { subscribe } = useWebSocket();

  const { isRunning, error, startJob, cancelJob } = useJob({
    start: (config: TrainingConfig) => api.startTraining(config),
    poll: (id: string) => api.trainingStatus(id),
    cancel: (id: string) => api.cancelTraining(id),
  });

  // Subscribe to epoch_complete WebSocket events
  useEffect(() => {
    const unsubscribe = subscribe("epoch_complete", (e: WSEvent) => {
      const data = e as unknown as EpochData;
      setEpochs((prev) => [...prev, data]);
      setLosses((prev) => [...prev, data.loss]);
      setAccuracies((prev) => [...prev, data.val_acc]);
    });
    return unsubscribe;
  }, [subscribe]);

  // Derive current training state from latest epoch
  const current = useMemo(() => {
    if (epochs.length === 0) return null;
    return epochs[epochs.length - 1];
  }, [epochs]);

  const handleStart = (values: Record<string, string | number | boolean>) => {
    // Reset state for new training run
    setEpochs([]);
    setLosses([]);
    setAccuracies([]);

    const config: TrainingConfig = {
      epochs_phase_a: Number(values.epochs_phase_a),
      epochs_phase_b: Number(values.epochs_phase_b),
      learning_rate: Number(values.learning_rate),
      batch_size: Number(values.batch_size),
      max_seq_length: Number(values.max_seq_length),
      num_classes: Number(values.num_classes),
      mlflow: Boolean(values.mlflow),
      experiment_name: String(values.experiment_name),
    };
    void startJob(config);
  };

  const totalEpochs = useMemo(() => {
    if (epochs.length === 0) return 0;
    // Infer from the field defaults or from the phases seen
    const lastEpoch = epochs[epochs.length - 1];
    return lastEpoch.epoch;
  }, [epochs]);

  return (
    <div className="flex h-full">
      {/* Main content area */}
      <div
        className="flex-1 overflow-auto p-8"
        style={{ marginRight: configOpen ? "20rem" : 0, transition: "margin-right 300ms ease-in-out" }}
      >
        {/* Page heading */}
        <h2
          className="text-xl mb-6"
          style={{ fontFamily: "var(--font-heading)", color: "var(--text-primary)" }}
        >
          Training
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

        {/* Progress header */}
        <div
          className="rounded p-5 mb-6 surface-grid"
          style={{
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border)",
          }}
        >
          <div className="flex items-center gap-6 flex-wrap">
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
                {isRunning ? "TRAINING" : epochs.length > 0 ? "COMPLETED" : "IDLE"}
              </span>
            </div>

            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Phase
              </span>
              <span
                className="text-sm font-bold"
                style={{ fontFamily: "var(--font-code)", color: "var(--data)" }}
              >
                {current ? current.phase : "--"}
              </span>
            </div>

            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Epoch
              </span>
              <span
                className="text-sm font-bold"
                style={{ fontFamily: "var(--font-code)", color: "var(--text-primary)" }}
              >
                {current ? totalEpochs : "--"}
              </span>
            </div>

            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Loss
              </span>
              <span
                className="text-sm font-bold"
                style={{ fontFamily: "var(--font-code)", color: "var(--threat)" }}
              >
                {current ? current.loss.toFixed(4) : "--"}
              </span>
            </div>

            <div>
              <span
                className="text-[10px] uppercase tracking-widest block mb-1"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Val Accuracy
              </span>
              <span
                className="text-sm font-bold"
                style={{ fontFamily: "var(--font-code)", color: "var(--safe)" }}
              >
                {current ? (current.val_acc * 100).toFixed(1) + "%" : "--"}
              </span>
            </div>
          </div>
        </div>

        {/* Sparkline charts */}
        {losses.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div
              className="rounded p-4"
              style={{
                backgroundColor: "var(--bg-surface)",
                border: "1px solid var(--border)",
              }}
            >
              <span
                className="text-[10px] uppercase tracking-widest block mb-2"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Loss Trend
              </span>
              <SparklineChart data={losses} color="var(--threat)" height={80} />
            </div>

            <div
              className="rounded p-4"
              style={{
                backgroundColor: "var(--bg-surface)",
                border: "1px solid var(--border)",
              }}
            >
              <span
                className="text-[10px] uppercase tracking-widest block mb-2"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Accuracy Trend
              </span>
              <SparklineChart data={accuracies} color="var(--safe)" height={80} />
            </div>
          </div>
        )}

        {/* Epoch table */}
        {epochs.length > 0 && (
          <div
            className="rounded overflow-hidden"
            style={{
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border)",
            }}
          >
            <div className="px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <span
                className="text-[10px] uppercase tracking-widest"
                style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
              >
                Epoch History
              </span>
            </div>
            <div className="overflow-auto" style={{ maxHeight: "400px" }}>
              <table className="w-full" style={{ fontFamily: "var(--font-code)" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    {["Epoch", "Phase", "Loss", "Train Acc", "Val Acc", "F1", "Time"].map(
                      (header) => (
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
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {epochs.map((row, i) => (
                    <tr
                      key={row.epoch}
                      style={{
                        borderBottom: "1px solid var(--border)",
                        opacity: i % 2 === 0 ? 1 : 0.75,
                      }}
                    >
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--text-primary)" }}>
                        {row.epoch}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--data)" }}>
                        {row.phase}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--threat)" }}>
                        {row.loss.toFixed(4)}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--text-primary)" }}>
                        {(row.train_acc * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--safe)" }}>
                        {(row.val_acc * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--purple)" }}>
                        {row.f1.toFixed(3)}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--text-muted)" }}>
                        {row.elapsed.toFixed(1)}s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isRunning && epochs.length === 0 && (
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
              NO TRAINING DATA
            </span>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              Configure parameters and start a training run
            </span>
          </div>
        )}
      </div>

      {/* Config panel */}
      <ConfigPanel
        title="Training Config"
        fields={trainingFields}
        onStart={handleStart}
        onCancel={() => void cancelJob()}
        isRunning={isRunning}
        isOpen={configOpen}
        onToggle={() => setConfigOpen((prev) => !prev)}
      />
    </div>
  );
}
