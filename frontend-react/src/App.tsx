import React from 'react';
import { BenchmarkProvider, useBenchmark } from './context/BenchmarkContext';
import { IconRail } from './components/workbench/IconRail';
import { StatusStrip } from './components/workbench/StatusStrip';
import { SlideDrawer } from './components/workbench/SlideDrawer';
import { BenchmarkCanvas } from './canvases/BenchmarkCanvas';
import { ManifoldCanvas } from './canvases/ManifoldCanvas';
import { QueryCanvas } from './canvases/QueryCanvas';
import { DatasetCanvas } from './canvases/DatasetCanvas';
import styles from './App.module.css';

const WorkbenchStage: React.FC = () => {
  const { activeCanvas } = useBenchmark();

  const renderActiveCanvas = () => {
    switch (activeCanvas) {
      case 'benchmarks':
        return <BenchmarkCanvas />;
      case 'manifold':
        return <ManifoldCanvas />;
      case 'query':
        return <QueryCanvas />;
      case 'datasets':
        return <DatasetCanvas />;
      default:
        return <BenchmarkCanvas />;
    }
  };

  return (
    <div className={styles.workbenchShell}>
      {/* Upper Workspace Row */}
      <div className={styles.workspaceRow}>
        {/* Left 48px Tool Rail */}
        <IconRail />

        {/* Full-Screen Main Canvas Area */}
        <main className={styles.canvasArea} role="region" aria-label="Active Workbench Canvas">
          {renderActiveCanvas()}
        </main>

        {/* Slide-In Inspector Drawer (From Right Edge) */}
        <SlideDrawer />
      </div>

      {/* Bottom 26px Status Strip */}
      <StatusStrip />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <BenchmarkProvider>
      <WorkbenchStage />
    </BenchmarkProvider>
  );
};

export default App;
