import { useState, useEffect, useCallback } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import StatCard from "../components/StatCard.tsx";
import ActivityLog from "../components/ActivityLog.tsx";
import type { LogEntry } from "../components/ActivityLog.tsx";
import { useDashboard } from "../hooks/useDashboard.ts";
import { useWebSocket } from "../hooks/useWebSocket.ts";

function formatPercent(value: number | undefined): string {
  if (value === undefined || value === null) return "\u2014";
  return `${(value * 100).toFixed(1)}%`;
}

export default function Dashboard() {
  const { data, loading, error } = useDashboard();
  const { subscribe } = useWebSocket();

  const [liveF1, setLiveF1] = useState<number | undefined>(undefined);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([
    {
      text: "Wintermute v4.0.0 ready",
      level: "ok",
      timestamp: new Date().toLocaleTimeString("en-GB", { hour12: false }),
    },
  ]);

  const addLogEntry = useCallback(
    (text: string, level: LogEntry["level"] = "info") => {
      setLogEntries((prev) => [
        ...prev,
        {
          text,
          level,
          timestamp: new Date().toLocaleTimeString("en-GB", { hour12: false }),
        },
      ]);
    },
    [],
  );

  useEffect(() => {
    const unsub1 = subscribe("epoch_complete", (e) => {
      const f1 = e.f1 as number | undefined;
      if (f1 !== undefined) {
        setLiveF1(f1);
      }
    });

    const unsub2 = subscribe("activity_log", (e) => {
      const text = (e.message ?? e.text ?? "") as string;
      const level = (e.level ?? "info") as LogEntry["level"];
      addLogEntry(text, level);
    });

    return () => {
      unsub1();
      unsub2();
    };
  }, [subscribe, addLogEntry]);

  // Derive chart data from family_counts
  const chartData = data?.family_counts
    ? Object.entries(data.family_counts).map(([name, count]) => ({
        name,
        count,
      }))
    : [];

  const displayF1 = liveF1 !== undefined ? liveF1 : data?.f1;

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center h-full">
        <p
          className="text-sm"
          style={{ color: "var(--text-muted)", fontFamily: "var(--font-code)" }}
        >
          Loading dashboard...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 flex items-center justify-center h-full">
        <p
          className="text-sm"
          style={{ color: "var(--threat)", fontFamily: "var(--font-code)" }}
        >
          Error: {error}
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Stat cards row */}
      <div className="grid grid-cols-5 gap-4">
        <StatCard
          label="MODEL"
          value={data?.model_version ? `v${data.model_version}` : "\u2014"}
          subtitle="MalBERT + GAT + Fusion"
          color="var(--data)"
        />
        <StatCard
          label="CLEAN TPR"
          value={formatPercent(data?.accuracy)}
          color="var(--safe)"
        />
        <StatCard
          label="ADV. TPR"
          value={"\u2014"}
          subtitle="Phase 5 not trained"
          color="var(--warn)"
        />
        <StatCard
          label="MACRO F1"
          value={formatPercent(displayF1)}
          subtitle={"target \u2265 0.90"}
          color="var(--data)"
        />
        <StatCard
          label="VAULT"
          value={String(data?.vault_size ?? 0)}
          color="var(--purple)"
        />
      </div>

      {/* Bottom section: chart + activity log */}
      <div className="grid grid-cols-3 gap-6">
        {/* Family distribution bar chart (2/3 width) */}
        <div
          className="col-span-2 rounded-lg p-4 surface-grid"
          style={{
            backgroundColor: "var(--bg-elevated)",
            border: "1px solid var(--border)",
          }}
        >
          <p
            className="text-[10px] uppercase tracking-widest mb-4"
            style={{
              fontFamily: "var(--font-heading)",
              color: "var(--text-muted)",
            }}
          >
            Family Distribution
          </p>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={chartData}
                margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
              >
                <XAxis
                  dataKey="name"
                  tick={{
                    fontSize: 10,
                    fill: "var(--text-muted)",
                    fontFamily: "var(--font-code)",
                  }}
                  axisLine={{ stroke: "var(--border)" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{
                    fontSize: 10,
                    fill: "var(--text-muted)",
                    fontFamily: "var(--font-code)",
                  }}
                  axisLine={{ stroke: "var(--border)" }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--bg-surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    fontFamily: "var(--font-code)",
                    fontSize: 11,
                    color: "var(--text-primary)",
                  }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {chartData.map((_entry, index) => (
                    <Cell key={index} fill="var(--data)" />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div
              className="flex items-center justify-center"
              style={{ height: 280 }}
            >
              <p
                className="text-xs"
                style={{
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-code)",
                }}
              >
                No family distribution data available.
              </p>
            </div>
          )}
        </div>

        {/* Activity log (1/3 width) */}
        <div className="col-span-1">
          <p
            className="text-[10px] uppercase tracking-widest mb-4"
            style={{
              fontFamily: "var(--font-heading)",
              color: "var(--text-muted)",
            }}
          >
            Activity Log
          </p>
          <ActivityLog entries={logEntries} maxHeight="310px" />
        </div>
      </div>
    </div>
  );
}
