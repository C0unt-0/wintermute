import { useState, useCallback, useRef, useEffect } from "react";

interface UseJobOptions<TConfig, TStatus> {
  start: (config: TConfig) => Promise<{ job_id: string }>;
  poll: (id: string) => Promise<TStatus>;
  cancel: (id: string) => Promise<unknown>;
  pollInterval?: number;
}

export function useJob<TConfig, TStatus extends { status: string }>({
  start,
  poll,
  cancel,
  pollInterval = 2000,
}: UseJobOptions<TConfig, TStatus>) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<TStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const startJob = useCallback(
    async (config: TConfig) => {
      try {
        setError(null);
        const res = await start(config);
        setJobId(res.job_id);
        setIsRunning(true);

        // Start polling
        intervalRef.current = setInterval(async () => {
          try {
            const s = await poll(res.job_id);
            setStatus(s);
            if (
              s.status === "COMPLETED" ||
              s.status === "FAILED" ||
              s.status === "CANCELLED"
            ) {
              stopPolling();
              setIsRunning(false);
            }
          } catch (e) {
            setError(e instanceof Error ? e.message : "Poll failed");
          }
        }, pollInterval);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Start failed");
        setIsRunning(false);
      }
    },
    [start, poll, pollInterval, stopPolling],
  );

  const cancelJob = useCallback(async () => {
    if (jobId) {
      try {
        await cancel(jobId);
        stopPolling();
        setIsRunning(false);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Cancel failed");
      }
    }
  }, [jobId, cancel, stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return { jobId, status, error, isRunning, startJob, cancelJob };
}
