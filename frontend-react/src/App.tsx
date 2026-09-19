import React from 'react';
import { BenchmarkProvider, useBenchmark } from './context/BenchmarkContext';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { OverviewView } from './views/OverviewView';
import { ViewPlaceholder } from './views/ViewPlaceholder';
import styles from './App.module.css';

const MainContent: React.FC = () => {
  const { activeView } = useBenchmark();

  const renderView = () => {
    switch (activeView) {
      case 'overview':
        return <OverviewView />;
      case 'datasets':
        return (
          <ViewPlaceholder
            viewId="datasets"
            title="Datasets: Empirical Signal Profiles & Benchmarking Corridors"
            isDataCritical
          />
        );
      case 'benchmarks':
        return (
          <ViewPlaceholder
            viewId="benchmarks"
            title="Benchmarks: Dual Pareto Frontiers & Recall vs. QPS"
            isDataCritical
          />
        );
      case 'experiments':
        return (
          <ViewPlaceholder
            viewId="experiments"
            title="Experiments: 6-Step Ablation Matrix & LaTeX Exporters"
            isDataCritical
          />
        );
      case 'index-builder':
        return (
          <ViewPlaceholder
            viewId="index-builder"
            title="Index Builder: Live Continuous Policy Configuration"
          />
        );
      case 'graph-explorer':
        return (
          <ViewPlaceholder
            viewId="graph-explorer"
            title="Graph Explorer: Concentric Multi-Layer HNSW Projection"
          />
        );
      case 'query-lab':
        return (
          <ViewPlaceholder
            viewId="query-lab"
            title="Query Lab: Real-Time Vector Traversal & Stagnation Exit"
          />
        );
      case 'node-analysis':
        return (
          <ViewPlaceholder
            viewId="node-analysis"
            title="Node Analysis: Intrinsic Dimensionality & Hubness"
          />
        );
      case 'system-metrics':
        return (
          <ViewPlaceholder
            viewId="system-metrics"
            title="System Metrics: Hardware Environment & Execution Profiling"
          />
        );
      case 'settings':
        return (
          <ViewPlaceholder
            viewId="settings"
            title="Settings: Interface Configuration & Telemetry"
          />
        );
      default:
        return <OverviewView />;
    }
  };

  return (
    <div className={styles.appShell}>
      <Sidebar />
      <div className={styles.mainLayout}>
        <Header />
        <main className={styles.contentArea}>
          {renderView()}
        </main>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <BenchmarkProvider>
      <MainContent />
    </BenchmarkProvider>
  );
};

export default App;
