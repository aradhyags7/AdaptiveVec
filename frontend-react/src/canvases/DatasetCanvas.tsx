/* Corpus Registry & Dimensionality Matrix Stage */
import React from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { motion, useReducedMotion } from 'framer-motion';
import type { Transition } from 'framer-motion';
import { CheckCircle2 } from 'lucide-react';
import styles from './CanvasShared.module.css';

export const DatasetCanvas: React.FC = () => {
  const { hardware } = useBenchmark();
  const shouldReduceMotion = useReducedMotion();

  const getEntranceProps = (idx: number) => ({
    initial: shouldReduceMotion ? false : { opacity: 0, y: 8 },
    animate: { opacity: 1, y: 0 },
    transition: (shouldReduceMotion
      ? { duration: 0 }
      : { type: 'spring', stiffness: 480, damping: 38, delay: idx * 0.035 }) as Transition
  });

  return (
    <div className={styles.canvasContainer}>
      {/* Precision Workspace Bar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <span className="text-xs font-semibold text-primary">Corpus Registry & Dimensionality Matrix</span>
          <span className={styles.barSep}>/</span>
          <span className="badge-pill emerald">
            <CheckCircle2 size={11} />
            <span>4 / 4 Corpora Evaluated (100% Real Empirical Data)</span>
          </span>
          <span className="badge-pill accent">SIFT-500K • 64-D Topology • 384-D NLP • Dynamic Churn</span>
        </div>

        <div className={styles.barRight}>
          <span className="text-xs text-muted tabular-nums">
            Evaluated Hardware: {hardware.model} ({hardware.cores}C/{hardware.threads}T)
          </span>
        </div>
      </div>

      {/* 4-Quadrant Corpus Matrix */}
      <div className={styles.datasetMatrixStage}>
        {/* Quadrant 1: SIFT-100K, SIFT-250K & SIFT-500K (Active Scale Testbed) */}
        <motion.div
          {...getEntranceProps(0)}
          className={`${styles.datasetQuad} ${styles.datasetQuadActive}`}
        >
          <div className={styles.datasetQuadHeader}>
            <div className="flex items-center gap-2">
              <span className={styles.datasetQuadTitle}>SIFT-100K, 250K & 500K Scale Expansion</span>
            </div>
            <span className="badge-pill emerald">
              <CheckCircle2 size={11} />
              <span>Active • Evaluated to 500K</span>
            </span>
          </div>

          <p className="text-xs text-secondary">
            Standard computer-vision local image feature descriptors (128-D FP32). Scaled 5.0× from 100K up to 500,000 base vectors on local Intel Core 5 210H hardware to validate algorithm scaling properties.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Evaluated Scales:</span>
              <span className={styles.specVal}>100K, 250K & 500,000 Vectors</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Dimensionality:</span>
              <span className={styles.specVal}>128-D FP32 (Euclidean L2)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>500K Graph Edges:</span>
              <span className={`${styles.specVal} text-emerald`}>12,147,532 (-9.44% / -1.27M links)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>500K Distance Hops:</span>
              <span className={`${styles.specVal} text-accent-glow`}>864.1 / query (-34.2% evals)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>500K Index RAM:</span>
              <span className={`${styles.specVal} text-emerald`}>294.3 MB (vs 299.1 MB Baseline)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>500K Recall@10:</span>
              <span className={`${styles.specVal} text-accent-glow`}>0.9183 (Fast Early Exit)</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Empirical Scaling Sweep: 100K &rarr; 250K &rarr; 500K complete &bull; Hardware: {hardware.model} (12 Threads, AVX2/FMA)
          </div>
        </motion.div>

        {/* Quadrant 2: Synthetic-Multi-Cluster */}
        <motion.div
          {...getEntranceProps(1)}
          className={styles.datasetQuad}
        >
          <div className={styles.datasetQuadHeader}>
            <span className={styles.datasetQuadTitle}>Synthetic-Multi-Cluster</span>
            <span className="badge-pill emerald">
              <CheckCircle2 size={11} />
              <span>Evaluated</span>
            </span>
          </div>

          <p className="text-xs text-secondary">
            Controlled Gaussian mixture manifold with 8 dense clusters connected by sparse bridge regions to stress adaptive degree allocation and hubness avoidance.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Base Vectors / Queries:</span>
              <span className={styles.specVal}>50,000 / 1,000</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Dimensionality:</span>
              <span className={styles.specVal}>64-D FP32</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Geometry Model:</span>
              <span className={styles.specVal}>8 Gaussian Centroids + Bridge Noise</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Local LID Spread:</span>
              <span className={`${styles.specVal} text-accent`}>8.4 &ndash; 28.6</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Peak Throughput:</span>
              <span className={`${styles.specVal} text-accent-glow`}>7,420.7 QPS (+50.45%)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>RAM Footprint:</span>
              <span className={`${styles.specVal} text-emerald`}>8.37 MB (Step 6 SQ8)</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Validation: Confirms hubness mitigation prevents premature disconnectivity.
          </div>
        </motion.div>

        {/* Quadrant 3: MiniLM-50K-MultiDomain (Sentence Transformers, 384D) */}
        <motion.div
          {...getEntranceProps(2)}
          className={styles.datasetQuad}
        >
          <div className={styles.datasetQuadHeader}>
            <span className={styles.datasetQuadTitle}>MiniLM-50K-MultiDomain (Sentence Transformers)</span>
            <span className="badge-pill emerald">
              <CheckCircle2 size={11} />
              <span>Active • Frontier 4 Evaluated</span>
            </span>
          </div>

          <p className="text-xs text-secondary">
            Dense 384-D semantic text embeddings (all-MiniLM-L6-v2) across 5 domains: SciFact, NFCorpus, ArguAna, Webmasters, and Android. Evaluated with 4-bit sub-byte quantization (SQ4) and dynamic ef.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Base Vectors / Queries:</span>
              <span className={styles.specVal}>57,893 / 1,000</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Dimensionality:</span>
              <span className={styles.specVal}>384-D FP32 (L2 Metric Space)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>RAM Footprint:</span>
              <span className={`${styles.specVal} text-emerald`}>16.69 MB (-81.7% SQ4 Total RAM)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Peak Throughput:</span>
              <span className={`${styles.specVal} text-accent-glow`}>54,624.8 QPS (+86.2% vs FP32)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Latency (p50 / p95):</span>
              <span className={styles.specVal}>160.2 &mu;s / 369.6 &mu;s (SQ4)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Recall Fidelity:</span>
              <span className={`${styles.specVal} text-emerald`}>0.9219 Peak &bull; 0.8514 Dyn-ef (+1.73%)</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Empirical 384D NLP benchmark complete &bull; Hardware: {hardware.model} (AVX2 SIMD nibble interleaving)
          </div>
        </motion.div>

        {/* Quadrant 4: SIFT-50K Dynamic Streaming Deletions & Healing */}
        <motion.div
          {...getEntranceProps(3)}
          className={styles.datasetQuad}
        >
          <div className={styles.datasetQuadHeader}>
            <span className={styles.datasetQuadTitle}>SIFT-50K Dynamic Streaming Deletions & Healing</span>
            <span className="badge-pill emerald">
              <CheckCircle2 size={11} />
              <span>Active • Frontier 2 Evaluated</span>
            </span>
          </div>

          <p className="text-xs text-secondary">
            Continuous streaming churn evaluation from 0% to 50% node deletions. Localized 2-hop bridge healing prevents graph disconnectivity, while zero-downtime compaction reclaims freed memory.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Active Vectors / Churn:</span>
              <span className={styles.specVal}>50,000 &rarr; 25,000 (0% &ndash; 50% Churn)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Dimensionality:</span>
              <span className={styles.specVal}>128-D FP32 (SIFT Descriptors)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>50% Churn Recall:</span>
              <span className={`${styles.specVal} text-emerald`}>0.9723 vs 0.9610 (+1.13% vs Tombstones)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Healing Throughput:</span>
              <span className={`${styles.specVal} text-accent-glow`}>54.2 &ndash; 73.4 &mu;s/op (&gt;13,500 dels/s)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Memory Reclaimed:</span>
              <span className={`${styles.specVal} text-emerald`}>-50.3% Index RAM (29.04 &rarr; 14.43 MB)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Compaction Speedup:</span>
              <span className={`${styles.specVal} text-accent-glow`}>62.1 ms (46.0&times; faster than 2,858 ms rebuild)</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Empirical Dynamic Churn Sweep complete &bull; Hardware: {hardware.model} (Striped Mutex Healing)
          </div>
        </motion.div>
      </div>
    </div>
  );
};
