export interface BenchmarkResult {
  dataset: string;
  configuration: string;
  n_samples: number;
  dim: number;
  n_queries: number;
  build_time_sec: number;
  total_edges: number;
  layer0_edges: number;
  upper_layer_edges: number;
  edge_change_pct: number;
  build_speedup_pct: number;
  memory_mb: number;
  recall_at_10: number;
  qps: number;
  distance_evaluations_per_query: number;
}

export interface BenchmarkDataFile {
  benchmark_results: BenchmarkResult[];
}

export type CanvasType = 'benchmarks' | 'manifold' | 'query' | 'datasets';
export type DrawerType = 'inspector' | 'data-export' | 'index-builder' | 'settings' | null;
export type ThemeMode = 'light' | 'dark';

export interface HardwareSpec {
  model: string;
  cores: number;
  threads: number;
  instructionSet: string;
  ramGb: number;
}

export const REAL_TESTBED_HARDWARE: HardwareSpec = {
  model: "Intel Core 5 210H",
  cores: 8,
  threads: 12,
  instructionSet: "AVX2 / FMA / FP32",
  ramGb: 16
};
