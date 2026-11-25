import React, { useState, useEffect, useCallback } from 'react';
import type { BenchmarkResult, CanvasType, DrawerType, ThemeMode } from '../types/benchmark';
import { REAL_TESTBED_HARDWARE } from '../types/benchmark';
import benchmarkDataRaw from '../data/benchmark_results.json';
import { BenchmarkContext } from './BenchmarkContextObject';

export const BenchmarkProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [results] = useState<BenchmarkResult[]>(benchmarkDataRaw.benchmark_results);
  const [activeCanvas, setActiveCanvas] = useState<CanvasType>('benchmarks');
  const [activeDrawer, setActiveDrawer] = useState<DrawerType>(null);
  
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

  // Global keyboard shortcuts (e.g. Esc to close drawer, 1-4 to switch canvases)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key === 'Escape') {
        setActiveDrawer(null);
      } else if (e.key === '1') {
        setActiveCanvas('benchmarks');
      } else if (e.key === '2') {
        setActiveCanvas('manifold');
      } else if (e.key === '3') {
        setActiveCanvas('query');
      } else if (e.key === '4') {
        setActiveCanvas('datasets');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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

  const toggleDrawer = useCallback((drawer: NonNullable<DrawerType>) => {
    setActiveDrawer(prev => (prev === drawer ? null : drawer));
  }, []);

  const closeDrawer = useCallback(() => {
    setActiveDrawer(null);
  }, []);

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
        activeCanvas,
        setActiveCanvas,
        activeDrawer,
        setActiveDrawer,
        toggleDrawer,
        closeDrawer,
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
