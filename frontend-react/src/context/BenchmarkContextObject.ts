import { createContext } from 'react';
import type { BenchmarkResult, CanvasType, DrawerType, ThemeMode, HardwareSpec } from '../types/benchmark';

export interface BenchmarkContextType {
  results: BenchmarkResult[];
  siftResults: BenchmarkResult[];
  syntheticResults: BenchmarkResult[];
  step5AdaEf: BenchmarkResult | undefined;
  step6Sq8: BenchmarkResult | undefined;
  baselineHnsw: BenchmarkResult | undefined;
  step4Hubness: BenchmarkResult | undefined;
  activeCanvas: CanvasType;
  setActiveCanvas: (canvas: CanvasType) => void;
  activeDrawer: DrawerType;
  setActiveDrawer: (drawer: DrawerType) => void;
  toggleDrawer: (drawer: NonNullable<DrawerType>) => void;
  closeDrawer: () => void;
  theme: ThemeMode;
  toggleTheme: () => void;
  audioEnabled: boolean;
  setAudioEnabled: (enabled: boolean) => void;
  hardware: HardwareSpec;
  serverStatus: 'online' | 'offline' | 'checking';
}

export const BenchmarkContext = createContext<BenchmarkContextType | undefined>(undefined);
