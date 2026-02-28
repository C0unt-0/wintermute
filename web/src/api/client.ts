const BASE = "/api/v1";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

// Type definitions
export interface DashboardData {
  model_version: string;
  f1: number;
  accuracy: number;
  vault_size: number;
  family_counts: Record<string, number>;
}

export interface JobResponse {
  job_id: string;
  poll_url: string;
}

export interface TrainingConfig {
  epochs_phase_a?: number;
  epochs_phase_b?: number;
  learning_rate?: number;
  batch_size?: number;
  max_seq_length?: number;
  num_classes?: number;
  mlflow?: boolean;
  experiment_name?: string;
}

export interface TrainingStatus {
  job_id: string;
  status: string;
  epoch: number;
  phase: string;
  loss: number;
  train_acc: number;
  val_acc: number;
  f1: number;
}

export interface AdversarialConfig {
  cycles?: number;
  episodes_per_cycle?: number;
  trades_beta?: number;
  ewc_lambda?: number;
  ppo_lr?: number;
  ppo_epochs?: number;
}

export interface AdversarialStatus {
  job_id: string;
  status: string;
  cycle: number;
  evasion_rate: number;
  adv_tpr: number;
  vault_size: number;
}

export interface PipelineConfig {
  data_dir?: string;
  max_seq_length?: number;
  n_samples?: number;
  output_dir?: string;
  seed?: number;
  epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  mask_prob?: number;
}

export interface PipelineStatus {
  job_id: string;
  status: string;
  operation: string;
  progress: number;
  message: string;
}

export interface VaultSample {
  id: string;
  family: string;
  confidence: number;
  mutations: number;
  cycle: number;
}

export interface VaultSampleDetail extends VaultSample {
  original_bytes: string;
  mutated_bytes: string;
  diff: string;
}

export const api = {
  // Dashboard
  dashboard: () => fetchJSON<DashboardData>("/dashboard"),

  // Scan
  scan: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/scan`, { method: "POST", body: form }).then(
      (r) => r.json(),
    ) as Promise<JobResponse>;
  },
  scanStatus: (id: string) =>
    fetchJSON<{
      job_id: string;
      status: string;
      result?: Record<string, unknown>;
    }>(`/status/${id}`),

  // Training
  startTraining: (config: TrainingConfig) =>
    fetchJSON<JobResponse>("/training/start", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  trainingStatus: (id: string) =>
    fetchJSON<TrainingStatus>(`/training/${id}/status`),
  cancelTraining: (id: string) =>
    fetchJSON<{ status: string }>(`/training/${id}/cancel`, {
      method: "POST",
    }),

  // Adversarial
  startAdversarial: (config: AdversarialConfig) =>
    fetchJSON<JobResponse>("/adversarial/start", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  adversarialStatus: (id: string) =>
    fetchJSON<AdversarialStatus>(`/adversarial/${id}/status`),
  cancelAdversarial: (id: string) =>
    fetchJSON<{ status: string }>(`/adversarial/${id}/cancel`, {
      method: "POST",
    }),

  // Pipeline
  startPipeline: (operation: string, config: PipelineConfig) =>
    fetchJSON<JobResponse>(`/pipeline/${operation}`, {
      method: "POST",
      body: JSON.stringify(config),
    }),
  pipelineStatus: (id: string) =>
    fetchJSON<PipelineStatus>(`/pipeline/${id}/status`),
  cancelPipeline: (id: string) =>
    fetchJSON<{ status: string }>(`/pipeline/${id}/cancel`, {
      method: "POST",
    }),

  // Vault
  vaultSamples: () => fetchJSON<VaultSample[]>("/vault/samples"),
  vaultSample: (id: string) =>
    fetchJSON<VaultSampleDetail>(`/vault/samples/${id}`),
};
