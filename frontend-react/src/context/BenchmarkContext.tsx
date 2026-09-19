import React, { createContext, useContext, useState, useEffect } from 'react';
import type { BenchmarkResult, ViewType, ThemeMode, HardwareSpec } from '../types/benchmark';
import { REAL_TESTBED_HARDWARE } from '../types/benchmark';
import benchmarkDataRaw from '../data/benchmark_results.json';

interface BenchmarkContextType {
  results: BenchmarkResult[];
  siftResults: BenchmarkResult[];
  syntheticResults: BenchmarkResult[];
  step5AdaEf: BenchmarkResult | undefined;
  step6Sq8: BenchmarkResult | undefined;
  baselineHnsw: BenchmarkResult | undefined;
  step4Hubness: BenchmarkResult | undefined;
  activeView: ViewType;
  setActiveView: (view: ViewType) => void;
  theme: ThemeMode;
  toggleTheme: () => void;
  audioEnabled: boolean;
  setAudioEnabled: (enabled: boolean) => void;
  hardware: HardwareSpec;
  serverStatus: 'online' | 'offline' | 'checking';
}

const BenchmarkContext = createContext<BenchmarkContextType | undefined>(undefined);

export const BenchmarkProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [results] = useState<BenchmarkResult[]>(benchmarkDataRaw.benchmark_results);
  const [activeView, setActiveView] = useState<ViewType>('overview');
  
  // Theme state with localStorage & system preference detection
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem('adaptivevec_theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  const [audioEnabled, setAudioEnabled] = useState<boolean>(false);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline' | 'checking'>('checking');

  // Apply theme class and save to localStorage
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('adaptivevec_theme', theme);
  }, [theme]);

  // Ping server status
  useEffect(() => {
    let isMounted = true;
    const checkServer = async () => {
      try {
        const res = await fetch('/api/status');
        if (res.ok && isMounted) {
          setServerStatus('online');
        } else if (isMounted) {
          setServerStatus('offline');
        }
      } catch {
        if (isMounted) setServerStatus('offline');
      }
    };
    checkServer();
    const interval = setInterval(checkServer, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Pre-filter dataset subsets
  const siftResults = results.filter(r => r.dataset === 'SIFT-100K subset');
  const syntheticResults = results.filter(r => r.dataset === 'Synthetic-Multi-Cluster');

  const baselineHnsw = siftResults.find(r => r.configuration.includes('1. Baseline HNSW'));
  const step4Hubness = siftResults.find(r => r.configuration.includes('4. + Hubness Regulation'));
  const step5AdaEf = siftResults.find(r => r.configuration.includes('5. + Ada-ef Stagnation Exit'));
  const step6Sq8 = siftResults.find(r => r.configuration.includes('6. + Asymmetric INT8 SQ8'));

  return (
    <BenchmarkContext.Provider
      value={{
        results,
        siftResults,
        syntheticResults,
        step5AdaEf,
        step6Sq8,
        baselineHnsw,
        step4Hubness,
        activeView,
        setActiveView,
        theme,
        toggleTheme,
        audioEnabled,
        setAudioEnabled,
        hardware: REAL_TESTBED_HARDWARE,
        serverStatus
      }}
    >
      {children}
    </BenchmarkContext.Provider>
  );
};

export const useBenchmark = (): BenchmarkContextType => {
  const ctx = useContext(BenchmarkContext);
  if (!ctx) {
    throw new Error('useBenchmark must be used within a BenchmarkProvider');
  }
  return ctx;
};
