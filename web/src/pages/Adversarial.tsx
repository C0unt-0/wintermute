import { useState, useEffect, useMemo, useRef } from "react";
import ConfigPanel from "../components/ConfigPanel.tsx";
import type { FieldDef } from "../components/ConfigPanel.tsx";
import SparklineChart from "../components/SparklineChart.tsx";
import { useJob } from "../hooks/useJob.ts";
import { useWebSocket } from "../hooks/useWebSocket.ts";
import type { WSEvent } from "../hooks/useWebSocket.ts";
import { api } from "../api/client.ts";
import type { AdversarialConfig } from "../api/client.ts";

interface EpisodeStep {
  step: number;
  action: string;
  position: number;
  confidence: number;
  valid: boolean;
}

interface CycleData {
  cycle: number;
  evasion_rate: number;
  ppo_loss?: number;
  adv_tpr: number;
  vault_size: number;
}

const adversarialFields: FieldDef[] = [
  { name: "cycles", label: "Cycles", type: "number", default: 10 },
  { name: "episodes_per_cycle", label: "Episodes/Cycle", type: "number", default: 500 },
  { name: "trades_beta", label: "TRADES Beta", type: "number", default: 1.0 },
  { name: "ewc_lambda", label: "EWC Lambda", type: "number", default: 0.4 },
  { name: "ppo_lr", label: "PPO Learning Rate", type: "number", default: 0.0003 },
  { name: "ppo_epochs", label: "PPO Epochs", type: "number", default: 4 },
];

export default function Adversarial() {
  const [evasionRates, setEvasionRates] = useState<number[]>([]);
  const [confidences, setConfidences] = useState<number[]>([]);
  const [episodes, setEpisodes] = useState<EpisodeStep[]>([]);
  const [cycles, setCycles] = useState<CycleData[]>([]);
  const [configOpen, setConfigOpen] = useState(true);

  const episodeLogRef = useRef<HTMLDivElement>(null);
  const { subscribe } = useWebSocket();

  const { isRunning, error, startJob, cancelJob } = useJob({
    start: (config: AdversarialConfig) => api.startAdversarial(config),
    poll: (id: string) => api.adversarialStatus(id),
    cancel: (id: string) => api.cancelAdversarial(id),
  });

  // Subscribe to adversarial_episode_step WebSocket events
  useEffect(() => {
    const unsubscribe = subscribe("adversarial_episode_step", (e: WSEvent) => {
      const data = e as unknown as EpisodeStep;
      setEpisodes((prev) => [...prev.slice(-500), data]);
      setConfidences((prev) => [...prev.slice(-200), data.confidence]);
    });
    return unsubscribe;
  }, [subscribe]);

  // Subscribe to adversarial_cycle_end WebSocket events
  useEffect(() => {
    const unsubscribe = subscribe("adversarial_cycle_end", (e: WSEvent) => {
      const data = e as unknown as CycleData;
      setCycles((prev) => [...prev, data]);
      setEvasionRates((prev) => [...prev, data.evasion_rate]);
    });
    return unsubscribe;
  }, [subscribe]);

  // Auto-scroll episode log to bottom
  useEffect(() => {
    if (episodeLogRef.current) {
      episodeLogRef.current.scrollTop = episodeLogRef.current.scrollHeight;
    }
  }, [episodes]);

  // Derive latest cycle data
  const latestCycle = useMemo(() => {
    if (cycles.length === 0) return null;
    return cycles[cycles.length - 1];
  }, [cycles]);

  const handleStart = (values: Record<string, string | number | boolean>) => {
    // Reset state for new adversarial run
    setEpisodes([]);
    setCycles([]);
    setEvasionRates([]);
    setConfidences([]);

    const config: AdversarialConfig = {
      cycles: Number(values.cycles),
      episodes_per_cycle: Number(values.episodes_per_cycle),
      trades_beta: Number(values.trades_beta),
      ewc_lambda: Number(values.ewc_lambda),
      ppo_lr: Number(values.ppo_lr),
      ppo_epochs: Number(values.ppo_epochs),
    };
    void startJob(config);
  };

  const hasData = cycles.length > 0 || episodes.length > 0;

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
          Adversarial
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

        {/* Red Team / Blue Team Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* RED TEAM Card */}
          <div
            className="rounded p-5"
            style={{
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--threat)",
              borderLeft: "3px solid var(--threat)",
            }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span
                className="text-[10px] uppercase tracking-widest"
                style={{ fontFamily: "var(--font-heading)", color: "var(--threat)" }}
              >
                RED TEAM
              </span>
            </div>
            <div className="space-y-2">
              <div>
                <span
                  className="text-[10px] uppercase tracking-widest block mb-1"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Agent
                </span>
                <span
                  className="text-sm font-bold"
                  style={{ fontFamily: "var(--font-code)", color: "var(--text-primary)" }}
                >
                  PPO Agent
                </span>
              </div>
              <div>
                <span
                  className="text-[10px] uppercase tracking-widest block mb-1"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Evasion Rate
                </span>
                <span
                  className="text-lg font-bold"
                  style={{ fontFamily: "var(--font-code)", color: "var(--threat)" }}
                >
                  {latestCycle ? (latestCycle.evasion_rate * 100).toFixed(1) + "%" : "--"}
                </span>
              </div>
            </div>
          </div>

          {/* BLUE TEAM Card */}
          <div
            className="rounded p-5"
            style={{
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--data)",
              borderLeft: "3px solid var(--data)",
            }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span
                className="text-[10px] uppercase tracking-widest"
                style={{ fontFamily: "var(--font-heading)", color: "var(--data)" }}
              >
                BLUE TEAM
              </span>
            </div>
            <div className="space-y-2">
              <div>
                <span
                  className="text-[10px] uppercase tracking-widest block mb-1"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Defender
                </span>
                <span
                  className="text-sm font-bold"
                  style={{ fontFamily: "var(--font-code)", color: "var(--text-primary)" }}
                >
                  Wintermute &middot; TRADES + EWC
                </span>
              </div>
              <div>
                <span
                  className="text-[10px] uppercase tracking-widest block mb-1"
                  style={{ fontFamily: "var(--font-heading)", color: "var(--text-muted)" }}
                >
                  Adversarial TPR
                </span>
                <span
                  className="text-lg font-bold"
                  style={{ fontFamily: "var(--font-code)", color: "var(--data)" }}
                >
                  {latestCycle ? (latestCycle.adv_tpr * 100).toFixed(1) + "%" : "--"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Sparkline Charts */}
        {(evasionRates.length > 0 || confidences.length > 0) && (
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
                Evasion Rate
              </span>
              <SparklineChart data={evasionRates} color="var(--threat)" height={80} />
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
                Defender Confidence
              </span>
              <SparklineChart data={confidences} color="var(--data)" height={80} />
            </div>
          </div>
        )}

        {/* Episode Action Log */}
        {episodes.length > 0 && (
          <div
            className="rounded overflow-hidden mb-6"
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
                Episode Action Log
              </span>
            </div>
            <div
              ref={episodeLogRef}
              className="overflow-auto p-3 space-y-px"
              style={{ maxHeight: "280px", fontFamily: "var(--font-code)" }}
            >
              {episodes.map((ep, i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 px-2 py-1 rounded text-xs"
                  style={{
                    backgroundColor: i % 2 === 0 ? "transparent" : "var(--bg-elevated)",
                  }}
                >
                  <span
                    className="w-8 text-right shrink-0"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {ep.step}
                  </span>
                  <span
                    className="w-4 text-center shrink-0 font-bold"
                    style={{ color: ep.valid ? "var(--safe)" : "var(--threat)" }}
                  >
                    {ep.valid ? "\u2713" : "\u2717"}
                  </span>
                  <span
                    className="w-28 shrink-0"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {ep.action}
                  </span>
                  <span
                    className="w-12 text-right shrink-0"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {ep.position}
                  </span>
                  <span
                    className="w-14 text-right shrink-0"
                    style={{ color: "var(--purple)" }}
                  >
                    {ep.confidence.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Cycle Table */}
        {cycles.length > 0 && (
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
                Cycle History
              </span>
            </div>
            <div className="overflow-auto" style={{ maxHeight: "400px" }}>
              <table className="w-full" style={{ fontFamily: "var(--font-code)" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    {["Cycle", "Evasion %", "PPO Loss", "Adv TPR", "Vault Size"].map(
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
                  {cycles.map((row, i) => (
                    <tr
                      key={row.cycle}
                      style={{
                        borderBottom: "1px solid var(--border)",
                        opacity: i % 2 === 0 ? 1 : 0.75,
                      }}
                    >
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--text-primary)" }}>
                        {row.cycle}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--threat)" }}>
                        {(row.evasion_rate * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--text-muted)" }}>
                        {row.ppo_loss != null ? row.ppo_loss.toFixed(4) : "--"}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--data)" }}>
                        {(row.adv_tpr * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--purple)" }}>
                        {row.vault_size}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isRunning && !hasData && (
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
              NO ADVERSARIAL DATA
            </span>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              Configure parameters and start an adversarial training run
            </span>
          </div>
        )}
      </div>

      {/* Config panel */}
      <ConfigPanel
        title="Adversarial Config"
        fields={adversarialFields}
        onStart={handleStart}
        onCancel={() => void cancelJob()}
        isRunning={isRunning}
        isOpen={configOpen}
        onToggle={() => setConfigOpen((prev) => !prev)}
      />
    </div>
  );
}
