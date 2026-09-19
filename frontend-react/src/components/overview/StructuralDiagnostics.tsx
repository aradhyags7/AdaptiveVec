import React from 'react';
import styles from './StructuralDiagnostics.module.css';

export const StructuralDiagnostics: React.FC = () => {
  return (
    <div className={styles.diagCard}>
      {/* Card Header */}
      <div className={styles.cardHeader}>
        <h4 className={styles.cardTitle}>Local Structural Diagnostics</h4>
        <span className="badge-pill illustrative">N=100K evaluation</span>
      </div>

      {/* 1. Local Intrinsic Dimensionality (LID) Histogram */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>1. Local intrinsic dimensionality (LID)</span>
          <span className={`${styles.sectionMean} tabular-nums text-accent`}>Mean: 12.4</span>
        </div>
        <div className={styles.subStats}>
          <span>Min: 4.1</span>
          <span>P95: 27.8</span>
          <span>Max: 34.2</span>
        </div>

        {/* Dynamic Histogram Bars */}
        <div className={styles.histWrapper}>
          <div className={styles.histBar} style={{ height: '28%' }} title="LID ~6: 28%" />
          <div className={styles.histBar} style={{ height: '42%' }} title="LID ~9: 42%" />
          <div className={styles.histBar} style={{ height: '65%' }} title="LID ~12: 65%" />
          <div className={styles.histBar} style={{ height: '88%' }} title="LID ~15: 88%" />
          <div className={styles.histBar} style={{ height: '98%' }} title="LID ~18: 98%" />
          <div className={styles.histBar} style={{ height: '72%' }} title="LID ~21: 72%" />
          <div className={styles.histBar} style={{ height: '54%' }} title="LID ~24: 54%" />
          <div className={styles.histBar} style={{ height: '38%' }} title="LID ~27: 38%" />
          <div className={styles.histBar} style={{ height: '24%' }} title="LID ~30: 24%" />
          <div className={styles.histBar} style={{ height: '16%' }} title="LID ~33: 16%" />
          
          <div className={styles.p95MarkerLine} />
          <span className={styles.p95Label}>P95 (27.8)</span>
        </div>
      </div>

      {/* 2. Local Density (k-NN distance) */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>2. Local density (k-NN distance)</span>
          <span className={`${styles.sectionMean} tabular-nums text-accent`}>Mean: 0.084</span>
        </div>
        <div className={styles.subStats}>
          <span>Min: 0.012</span>
          <span>P95: 0.173</span>
          <span>Skew: +1.48</span>
        </div>

        {/* SVG Density Curve */}
        <div className={styles.densitySvgWrapper}>
          <svg viewBox="0 0 300 64" className={styles.densitySvg}>
            <defs>
              <linearGradient id="densityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.0" />
              </linearGradient>
            </defs>
            <path
              d="M 10 58 Q 50 55, 90 28 T 170 12 T 230 45 T 290 58 L 290 58 L 10 58 Z"
              fill="url(#densityGrad)"
            />
            <path
              d="M 10 58 Q 50 55, 90 28 T 170 12 T 230 45 T 290 58"
              fill="none"
              stroke="var(--accent)"
              strokeWidth="2"
            />
            <circle cx="170" cy="12" r="3.5" fill="var(--accent)" />
            <line x1="170" y1="12" x2="170" y2="58" stroke="var(--border-default)" strokeDasharray="3 3" />
          </svg>
        </div>
      </div>

      {/* 3. Adaptive M Allocation Breakdown */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>3. Capacity allocation (M &isin; [8, 24])</span>
          <span className={`${styles.sectionMean} tabular-nums`}>Mean M: 14.8</span>
        </div>
        <div className={styles.subStats}>
          <span>Target Mean: 16.0</span>
          <span>Std Dev: 4.2</span>
          <span>Pruning: -7.4%</span>
        </div>

        <div className={styles.mBarRows}>
          <div className={styles.mRow}>
            <span className={styles.mLabel}>M=8</span>
            <div className={styles.mTrack}><div className={styles.mFill} style={{ width: '14%' }} /></div>
            <span className={`${styles.mPct} tabular-nums`}>14%</span>
          </div>
          <div className={styles.mRow}>
            <span className={styles.mLabel}>M=12</span>
            <div className={styles.mTrack}><div className={styles.mFill} style={{ width: '24%' }} /></div>
            <span className={`${styles.mPct} tabular-nums`}>24%</span>
          </div>
          <div className={styles.mRow}>
            <span className={styles.mLabel}>M=16</span>
            <div className={styles.mTrack}><div className={styles.mFill} style={{ width: '38%' }} /></div>
            <span className={`${styles.mPct} tabular-nums`}>38%</span>
          </div>
          <div className={styles.mRow}>
            <span className={styles.mLabel}>M=20</span>
            <div className={styles.mTrack}><div className={styles.mFill} style={{ width: '16%' }} /></div>
            <span className={`${styles.mPct} tabular-nums`}>16%</span>
          </div>
          <div className={styles.mRow}>
            <span className={styles.mLabel}>M=24</span>
            <div className={styles.mTrack}><div className={styles.mFill} style={{ width: '8%' }} /></div>
            <span className={`${styles.mPct} tabular-nums`}>8%</span>
          </div>
        </div>
      </div>
    </div>
  );
};
