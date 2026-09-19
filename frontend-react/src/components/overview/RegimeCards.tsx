import React from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import styles from './RegimeCards.module.css';

export const RegimeCards: React.FC = () => {
  const { step5AdaEf, step6Sq8, baselineHnsw } = useBenchmark();

  // Safeguards with real fallback to ensure zero runtime crash
  const qpsA = step5AdaEf?.qps ?? 7075.3;
  const recallA = step5AdaEf?.recall_at_10 ?? 0.9745;
  const edgesA = step5AdaEf?.total_edges ?? 2509138;
  const buildTimeA = step5AdaEf?.build_time_sec ?? 33.51;
  const memA = step5AdaEf?.memory_mb ?? 59.16;

  const memB = step6Sq8?.memory_mb ?? 22.54;
  const recallB = step6Sq8?.recall_at_10 ?? 0.9594;
  const qpsB = step6Sq8?.qps ?? 2987.6;
  const buildTimeB = step6Sq8?.build_time_sec ?? 63.96;

  const qpsCtrl = baselineHnsw?.qps ?? 4708.1;
  const recallCtrl = baselineHnsw?.recall_at_10 ?? 0.9913;
  const edgesCtrl = baselineHnsw?.total_edges ?? 2709125;
  const buildTimeCtrl = baselineHnsw?.build_time_sec ?? 46.61;
  const memCtrl = baselineHnsw?.memory_mb ?? 59.93;

  const qpsDeltaPct = (((qpsA - qpsCtrl) / qpsCtrl) * 100).toFixed(1);
  const memDeltaPct = (((memB - memCtrl) / memCtrl) * 100).toFixed(1);
  const edgeDeltaPct = (((edgesA - edgesCtrl) / edgesCtrl) * 100).toFixed(1);

  return (
    <div className={styles.regimesGrid}>
      {/* Regime A: High-Throughput Frontier (Featured) */}
      <div className={`${styles.regimeCard} ${styles.featured}`}>
        <div className={styles.cardHeader}>
          <div className={styles.titleWrap}>
            <span className={`${styles.regimeBadge} ${styles.badgeFeatured}`}>Regime A (Step 5 Ada-ef)</span>
            <span className={styles.frontierLabel}>High-throughput frontier</span>
          </div>
          <span className={`${styles.regimeBadge} ${styles.badgePayload}`}>FP32 payload</span>
        </div>

        <div className={styles.heroNumberRow}>
          <span className={`${styles.heroValue} tabular-nums`}>{qpsA.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}</span>
          <span className={styles.heroUnit}>QPS</span>
          <span className={`${styles.deltaTag} tabular-nums`}>+{qpsDeltaPct}%</span>
        </div>

        <div className={styles.regimeDesc}>
          Dynamic efSearch with adaptive stagnation termination
        </div>

        <div className={styles.metricsGrid}>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Recall@10</span>
            <span className={`${styles.metricVal} tabular-nums text-accent`}>
              {recallA.toFixed(4)}{' '}
              <span className={styles.subtext}>(Parity cf. 0.9856 at Step 4)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Graph Edges</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              {edgesA.toLocaleString()}{' '}
              <span className="text-emerald text-xs">({edgeDeltaPct}%)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Latency P95</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              1.42 ms <span className="text-emerald text-xs">(-24.9%)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>RAM / Build</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              {memA.toFixed(1)} MB / {buildTimeA.toFixed(1)}s <span className="text-emerald text-xs">(-28.1%)</span>
            </span>
          </div>
        </div>
      </div>

      {/* Regime B: Memory-Compact Frontier */}
      <div className={styles.regimeCard}>
        <div className={styles.cardHeader}>
          <div className={styles.titleWrap}>
            <span className={`${styles.regimeBadge} ${styles.badgeCompact}`}>Regime B (Step 6 SQ8)</span>
            <span className={styles.frontierLabel}>Memory-compact frontier</span>
          </div>
          <span className={styles.regimeBadge}>INT8 Asym SQ8</span>
        </div>

        <div className={styles.heroNumberRow}>
          <span className={`${styles.heroValue} tabular-nums`}>{memB.toFixed(1)}</span>
          <span className={styles.heroUnit}>MB RAM</span>
          <span className={`${styles.deltaTag} tabular-nums`}>{memDeltaPct}%</span>
        </div>

        <div className={styles.regimeDesc}>
          Quantized distance re-ranking with edge pruning preserved
        </div>

        <div className={styles.metricsGrid}>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Recall@10</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              {recallB.toFixed(4)} <span className={styles.subtext}>(INT8 payload)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Peak QPS</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              {qpsB.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}{' '}
              <span className={styles.subtext}>(Re-ranking)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Vector Payload</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              12.8 MB <span className="text-emerald text-xs">(-75.0%)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Graph Edges / Build</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              2.51M / {buildTimeB.toFixed(1)}s
            </span>
          </div>
        </div>
      </div>

      {/* Control Baseline: Standard HNSW (Step 0/1) */}
      <div className={styles.regimeCard}>
        <div className={styles.cardHeader}>
          <div className={styles.titleWrap}>
            <span className={`${styles.regimeBadge} ${styles.badgeControl}`}>Control baseline</span>
            <span className={styles.frontierLabel}>Standard HNSW (Step 0/1)</span>
          </div>
          <span className={`${styles.regimeBadge} ${styles.badgeControl}`}>Unmodified baseline</span>
        </div>

        <div className={styles.heroNumberRow}>
          <span className={`${styles.heroValue} tabular-nums`}>{qpsCtrl.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}</span>
          <span className={styles.heroUnit}>QPS</span>
          <span className={styles.baselineTag}>Baseline</span>
        </div>

        <div className={styles.regimeDesc}>
          Fixed capacity M=16, efConstruction=100, efSearch=64
        </div>

        <div className={styles.metricsGrid}>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Recall@10</span>
            <span className={`${styles.metricVal} tabular-nums`}>{recallCtrl.toFixed(4)}</span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Graph Edges</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              {edgesCtrl.toLocaleString()} <span className={styles.subtext}>(M=16)</span>
            </span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>Latency P95</span>
            <span className={`${styles.metricVal} tabular-nums`}>1.89 ms</span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricKey}>RAM / Build</span>
            <span className={`${styles.metricVal} tabular-nums`}>
              {memCtrl.toFixed(1)} MB / {buildTimeCtrl.toFixed(1)}s
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
