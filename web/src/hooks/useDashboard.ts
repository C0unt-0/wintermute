import { useState, useEffect } from "react";
import { api } from "../api/client.ts";
import type { DashboardData } from "../api/client.ts";

export function useDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .dashboard()
      .then(setData)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "Failed to load dashboard"),
      )
      .finally(() => setLoading(false));
  }, []);

  return { data, loading, error };
}
