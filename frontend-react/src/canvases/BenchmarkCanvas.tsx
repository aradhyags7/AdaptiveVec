import React from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { FileCode } from 'lucide-react';
import styles from './CanvasShared.module.css';

export const BenchmarkCanvas: React.FC = () => {
  const { toggleDrawer } = useBenchmark();

  return (
    <div className={styles.canvasContainer}>
      {/* Top Thin Workspace Toolbar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <div className={styles.segmentedGroup}>
            <button className={`${styles.segBtn} ${styles.segBtnActive}`}>Dual Pareto Frontiers</button>
            <button className={styles.segBtn}>6-Step Ablation Matrix</button>
            <button className={styles.segBtn}>3-Regime Matrix</button>
          </div>
          <span className={styles.barSep}>/</span>
          <span className="badge-pill accent">SIFT-100K (128-D FP32)</span>
          <span className={styles.wireframeTag}>TODO — not yet wired</span>
        </div>

        <div className={styles.barRight}>
          <button
            className={styles.barActionBtn}
            onClick={() => toggleDrawer('data-export')}
            title="Open Raw Data & LaTeX Exporters (Cmd+D)"
          >
            <FileCode size={13} />
            <span>Raw Data & Exports</span>
            <kbd className={styles.kbd}>Cmd+D</kbd>
          </button>
        </div>
      </div>

      {/* Main Viewport Stage (No card boxes — scientific split panes) */}
      <div className={styles.stageGrid2}>
        {/* Left Pane: Dual Pareto Frontiers */}
        <div className={styles.panePrimary}>
          <div className={styles.paneHeader}>
            <span className={styles.paneTitle}>Dual Pareto Frontier Stage</span>
            <span className="text-xs text-muted">Recall@10 vs. Throughput (QPS)</span>
          </div>

          <div className={styles.wireframePlot}>
            <div className={styles.plotCrosshairX} />
            <div className={styles.plotCrosshairY} />

            <div className={styles.plotPoint} style={{ left: '78%', bottom: '74%' }}>
              <div className={styles.pointDotFeatured} />
              <span className={styles.pointLabel}>
                Step 5 Ada-ef (7,075.3 QPS, 0.9745 Recall)
              </span>
            </div>

            <div className={styles.plotPoint} style={{ left: '48%', bottom: '90%' }}>
              <div className={styles.pointDot} />
              <span className={styles.pointLabel}>
                Step 1 Baseline (4,708.1 QPS, 0.9913 Recall)
              </span>
            </div>

            <div className={styles.plotPoint} style={{ left: '30%', bottom: '60%' }}>
              <div className={styles.pointDot} />
              <span className={styles.pointLabel}>
                Step 6 SQ8 (2,987.6 QPS, 0.9594 Recall, 22.5 MB)
              </span>
            </div>

            <div className={styles.watermark}>
              <span>PRIMARY BENCHMARK WORKBENCH</span>
              <span className={styles.subWatermark}>Full-viewport stage • No card wall</span>
            </div>
          </div>
        </div>

        {/* Right Pane: 3-Regime Comparative Architecture */}
        <div className={styles.paneSecondary}>
          <div className={styles.paneHeader}>
            <span className={styles.paneTitle}>3-Regime Comparative Architecture</span>
            <span className="badge-pill accent">Empirical validation</span>
          </div>

          <div className={styles.regimeMatrixList}>
            {/* Regime A */}
            <div className={`${styles.regimeBlock} ${styles.blockFeatured}`}>
              <div className={styles.blockRow}>
                <span className={styles.blockBadge}>Regime A (Step 5 Ada-ef)</span>
                <span className="text-emerald tabular-nums font-semibold">+50.3% QPS</span>
              </div>
              <div className={styles.blockNumberRow}>
                <span className={`${styles.blockHeroNum} tabular-nums`}>7,075.3</span>
                <span className={styles.blockHeroUnit}>QPS</span>
              </div>
              <div className={styles.blockSub}>
                0.9745 Recall@10 (Parity cf. 0.9856 at Step 4) &bull; 2.51M edges (-7.4%)
              </div>
            </div>

            {/* Regime B */}
            <div className={styles.regimeBlock}>
              <div className={styles.blockRow}>
                <span className={styles.blockBadge}>Regime B (Step 6 SQ8)</span>
                <span className="text-emerald tabular-nums font-semibold">-62.4% RAM</span>
              </div>
              <div className={styles.blockNumberRow}>
                <span className={`${styles.blockHeroNum} tabular-nums`}>22.5</span>
                <span className={styles.blockHeroUnit}>MB RAM</span>
              </div>
              <div className={styles.blockSub}>
                0.9594 Recall@10 &bull; 2,987.6 QPS (Quantized re-ranking)
              </div>
            </div>

            {/* Control Baseline */}
            <div className={styles.regimeBlock}>
              <div className={styles.blockRow}>
                <span className={styles.blockBadge}>Control Baseline</span>
                <span className="text-muted text-xs">Standard HNSW (Step 0/1)</span>
              </div>
              <div className={styles.blockNumberRow}>
                <span className={`${styles.blockHeroNum} tabular-nums`}>4,708.1</span>
                <span className={styles.blockHeroUnit}>QPS</span>
              </div>
              <div className={styles.blockSub}>
                0.9913 Recall@10 &bull; 2.71M edges (M=16, efC=100)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
