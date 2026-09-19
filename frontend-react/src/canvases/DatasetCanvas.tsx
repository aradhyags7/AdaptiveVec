import React from 'react';
import styles from './CanvasShared.module.css';

export const DatasetCanvas: React.FC = () => {
  return (
    <div className={styles.canvasContainer}>
      {/* Top Thin Workspace Toolbar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <span className="text-xs font-semibold text-primary">Corpus Registry</span>
          <span className={styles.barSep}>/</span>
          <span className="badge-pill accent">SIFT-100K Active</span>
          <span className={styles.wireframeTag}>TODO — not yet wired</span>
        </div>

        <div className={styles.barRight}>
          <span className="text-xs text-muted tabular-nums">4 Target Corpora &bull; N=250,000 Total Vectors</span>
        </div>
      </div>

      {/* 4-Quadrant Corpus Matrix */}
      <div className={styles.stageGrid2}>
        {/* Quadrant 1: SIFT-100K Active */}
        <div className={styles.corpusCard}>
          <div className={styles.corpusHeader}>
            <span className={styles.corpusTitle}>SIFT-100K (128-D FP32)</span>
            <span className="badge-pill emerald">Active &bull; Fully Evaluated</span>
          </div>
          <div className={styles.corpusBody}>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Samples / Dimension:</span>
              <span className="tabular-nums">100,000 / 128-D</span>
            </div>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Metric Space:</span>
              <span>Euclidean (L2)</span>
            </div>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Mean Intrinsic Dim (LID):</span>
              <span className="tabular-nums text-accent">~9.8 (MLE)</span>
            </div>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Tested Ablations:</span>
              <span className="tabular-nums">Steps 1–6 (Full Protocol)</span>
            </div>
          </div>
        </div>

        {/* Quadrant 2: Synthetic-Multi-Cluster */}
        <div className={styles.corpusCard}>
          <div className={styles.corpusHeader}>
            <span className={styles.corpusTitle}>Synthetic-Multi-Cluster</span>
            <span className="badge-pill emerald">Evaluated</span>
          </div>
          <div className={styles.corpusBody}>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Samples / Dimension:</span>
              <span className="tabular-nums">50,000 / 64-D</span>
            </div>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Geometry:</span>
              <span>Gaussian Mixture with Bridge Noise</span>
            </div>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Tested Ablations:</span>
              <span className="tabular-nums">Steps 1–6</span>
            </div>
          </div>
        </div>

        {/* Quadrant 3: DBpedia-100K (CRITICAL: MUST BE NOT RUN) */}
        <div className={`${styles.corpusCard} ${styles.corpusCardUnrun}`}>
          <div className={styles.corpusHeader}>
            <span className={styles.corpusTitle}>DBpedia-100K (768-D OpenAI)</span>
            <span className="badge-pill amber">NOT RUN — PENDING EVALUATION</span>
          </div>
          <div className={styles.corpusBody}>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Evaluation Status:</span>
              <span className="text-amber font-medium">Pending testbed execution</span>
            </div>
            <div className={styles.unrunNotice}>
              Zero synthetic data. DBpedia-100K has not been evaluated on this hardware testbed. No metrics are displayed.
            </div>
          </div>
        </div>

        {/* Quadrant 4: GloVe-100 */}
        <div className={`${styles.corpusCard} ${styles.corpusCardUnrun}`}>
          <div className={styles.corpusHeader}>
            <span className={styles.corpusTitle}>GloVe-100 (100-D Angular)</span>
            <span className="badge-pill">Pending Evaluation</span>
          </div>
          <div className={styles.corpusBody}>
            <div className={styles.corpusRow}>
              <span className={styles.k}>Evaluation Status:</span>
              <span className="text-muted">Corpus queued for future benchmark sweep</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
