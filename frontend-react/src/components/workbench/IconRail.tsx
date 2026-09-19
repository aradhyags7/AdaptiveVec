import React from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import type { CanvasType } from '../../types/benchmark';
import {
  LineChart,
  Network,
  Compass,
  Database,
  FileCode,
  Hammer,
  Sliders,
  Sun,
  Moon,
  Zap
} from 'lucide-react';
import styles from './IconRail.module.css';

interface ToolItem {
  id: CanvasType;
  label: string;
  icon: React.ReactNode;
  shortcut: string;
}

const tools: ToolItem[] = [
  { id: 'benchmarks', label: 'Benchmark & Pareto Workbench (1)', icon: <LineChart size={18} />, shortcut: '1' },
  { id: 'manifold', label: 'Manifold & Graph Workbench (2)', icon: <Network size={18} />, shortcut: '2' },
  { id: 'query', label: 'Query Traversal Workbench (3)', icon: <Compass size={18} />, shortcut: '3' },
  { id: 'datasets', label: 'Dataset Corpus Workbench (4)', icon: <Database size={18} />, shortcut: '4' }
];

export const IconRail: React.FC = () => {
  const { activeCanvas, setActiveCanvas, toggleDrawer, activeDrawer, theme, toggleTheme } = useBenchmark();

  return (
    <nav className={styles.rail} aria-label="Workbench Tool Rail">
      {/* Brand Icon Mark */}
      <div className={styles.brandMark} title="AdaptiveVec Research Workbench">
        <Zap size={16} className={styles.zapIcon} />
      </div>

      {/* Primary Tool Switcher */}
      <div className={styles.toolGroup}>
        {tools.map(tool => {
          const isActive = activeCanvas === tool.id;
          return (
            <button
              key={tool.id}
              className={`${styles.railBtn} ${isActive ? styles.railBtnActive : ''}`}
              onClick={() => setActiveCanvas(tool.id)}
              title={tool.label}
              aria-label={tool.label}
              aria-pressed={isActive}
            >
              {tool.icon}
              {isActive && <div className={styles.activeIndicator} />}
            </button>
          );
        })}
      </div>

      {/* Spacer */}
      <div className={styles.spacer} />

      {/* Bottom Drawer & Utility Triggers */}
      <div className={styles.bottomGroup}>
        <button
          className={`${styles.railBtn} ${activeDrawer === 'data-export' ? styles.railBtnActive : ''}`}
          onClick={() => toggleDrawer('data-export')}
          title="Inspect Raw Data & Exports (Cmd+D)"
          aria-label="Raw Data & Exports"
        >
          <FileCode size={17} />
        </button>

        <button
          className={`${styles.railBtn} ${activeDrawer === 'index-builder' ? styles.railBtnActive : ''}`}
          onClick={() => toggleDrawer('index-builder')}
          title="Index Builder Policy Action Drawer"
          aria-label="Index Builder Policy Drawer"
        >
          <Hammer size={17} />
        </button>

        <button
          className={`${styles.railBtn} ${activeDrawer === 'settings' ? styles.railBtnActive : ''}`}
          onClick={() => toggleDrawer('settings')}
          title="Workbench Settings"
          aria-label="Workbench Settings"
        >
          <Sliders size={17} />
        </button>

        <button
          className={styles.railBtn}
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme'}
          aria-label="Toggle Theme"
        >
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </nav>
  );
};
