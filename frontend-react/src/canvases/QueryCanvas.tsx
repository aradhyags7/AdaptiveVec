import React from 'react';
import { Play } from 'lucide-react';
import styles from './CanvasShared.module.css';

export const QueryCanvas: React.FC = () => {
  return (
    <div className={styles.canvasContainer}>
      {/* Top Thin Workspace Toolbar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <div className={styles.segmentedGroup}>
            <button className={`${styles.segBtn} ${styles.segBtnActive}`}>Query #0</button>
            <button className={styles.segBtn}>Query #1</button>
            <button className={styles.segBtn}>Query #2</button>
          </div>
          <span className={styles.barSep}>/</span>
          <span className="badge-pill illustrative">Illustrative Demo</span>
          <span className={styles.wireframeTag}>TODO — not yet wired</span>
        </div>

        <div className={styles.barRight}>
          <button className={styles.barActionBtn}>
            <Play size={12} fill="currentColor" />
            <span>Trace Traversal</span>
          </button>
        </div>
      </div>

      {/* Main Viewport Stage */}
      <div className={styles.stageFull}>
        <div className={styles.queryStage}>
          <div className={styles.trajectoryHeader}>
            <span className={styles.trajectoryTitle}>
              Hop-by-Hop Traversal Convergence &bull; Stagnation Window (p=6, &epsilon;=10⁻⁴)
            </span>
            <span className="tabular-nums text-emerald font-medium">
              Early Exit: Hop 16 (vs. Hop 24 baseline) &bull; -44.9% Distance Evals
            </span>
          </div>

          <div className={styles.trajectoryPathMock}>
            <div className={styles.entryHop}>Entry #10,402 (Layer 4)</div>
            <div className={styles.hopArrow}>&rarr;</div>
            <div className={styles.midHop}>Hop 1 (#12,890)</div>
            <div className={styles.hopArrow}>&rarr;</div>
            <div className={styles.midHop}>Hop 2 (#34,011)</div>
            <div className={styles.hopArrow}>&rarr;</div>
            <div className={styles.exitHop}>Adaptive Exit #48,219 (Hop 16)</div>
          </div>

          <div className={styles.watermark}>
            <span>QUERY TRAVERSAL WORKBENCH</span>
            <span className={styles.subWatermark}>Full-viewport stage &bull; Labeled illustrative demonstration</span>
          </div>
        </div>
      </div>
    </div>
  );
};
