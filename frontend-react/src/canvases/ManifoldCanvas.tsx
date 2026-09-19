import React from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { Info } from 'lucide-react';
import styles from './CanvasShared.module.css';

export const ManifoldCanvas: React.FC = () => {
  const { toggleDrawer } = useBenchmark();

  return (
    <div className={styles.canvasContainer}>
      {/* Top Thin Workspace Toolbar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <div className={styles.segmentedGroup}>
            <button className={`${styles.segBtn} ${styles.segBtnActive}`}>UMAP</button>
            <button className={styles.segBtn}>PCA</button>
          </div>
          <span className={styles.barSep}>|</span>
          <div className={styles.segmentedGroup}>
            <button className={`${styles.segBtn} ${styles.segBtnActive}`}>Adaptive M</button>
            <button className={styles.segBtn}>LID</button>
            <button className={styles.segBtn}>Density</button>
          </div>
          <span className={styles.barSep}>/</span>
          <span className="text-xs text-muted">Metric: Cosine</span>
          <span className={styles.wireframeTag}>TODO — not yet wired</span>
        </div>

        <div className={styles.barRight}>
          <button
            className={styles.barActionBtn}
            onClick={() => toggleDrawer('inspector')}
            title="Toggle Node Inspector Drawer"
          >
            <Info size={13} />
            <span>Node Inspector</span>
            <kbd className={styles.kbd}>Cmd+I</kbd>
          </button>
        </div>
      </div>

      {/* Edge-to-Edge 2D Manifold Stage */}
      <div className={styles.stageFull}>
        <div className={styles.coordinateStage}>
          {/* Subtle Grid Lines */}
          <div className={styles.gridOverlay} />

          {/* Wireframe Mock Clusters */}
          <div className={styles.mockCluster1}>
            <div className={styles.clusterCore}>Dense Core (M=10)</div>
          </div>

          <div
            className={styles.mockSelectedNode}
            onClick={() => toggleDrawer('inspector')}
            title="Click to inspect node"
          >
            <div className={styles.pulseRing} />
            <div className={styles.nodePoint} />
            <span className={styles.nodeLabel}>Node #48,219 (LID=26.4, M=22) [Click to Inspect]</span>
          </div>

          <div className={styles.mockCluster2}>
            <div className={styles.clusterCore}>Manifold Crest (M=24)</div>
          </div>

          {/* Coordinate Rulers */}
          <div className={styles.rulerX}>Axis 1 [Primary Variance Spread] &rarr;</div>
          <div className={styles.rulerY}>Axis 2 [Manifold Curvature] &uarr;</div>

          <div className={styles.watermark}>
            <span>EDGE-TO-EDGE 2D MANIFOLD OBSERVATORY</span>
            <span className={styles.subWatermark}>Full-viewport stage &bull; Clicking any node triggers slide-in inspector</span>
          </div>
        </div>
      </div>
    </div>
  );
};
