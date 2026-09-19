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

import { AnimatePresence, motion } from 'framer-motion';

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

        {/* Full-Screen Main Canvas Area with Physical Cross-Fade */}
        <main className={styles.canvasArea} role="region" aria-label="Active Workbench Canvas">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeCanvas}
              initial={{ opacity: 0, scale: 0.994 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.996 }}
              transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
              style={{ width: '100%', height: '100%' }}
            >
              {renderActiveCanvas()}
            </motion.div>
          </AnimatePresence>
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
