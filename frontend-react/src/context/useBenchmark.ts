import { useContext } from 'react';
import { BenchmarkContext } from './BenchmarkContextObject';
import type { BenchmarkContextType } from './BenchmarkContextObject';

export const useBenchmark = (): BenchmarkContextType => {
  const ctx = useContext(BenchmarkContext);
  if (!ctx) {
    throw new Error('useBenchmark must be used within a BenchmarkProvider');
  }
  return ctx;
};
