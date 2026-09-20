import React from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { motion, useReducedMotion } from 'framer-motion';
import type { Transition } from 'framer-motion';
import { AlertTriangle, CheckCircle2, Clock } from 'lucide-react';
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
          <span className="badge-pill accent">SIFT-100K Active</span>
          <span className="badge-pill">4 Target Corpora</span>
        </div>

        <div className={styles.barRight}>
          <span className="text-xs text-muted tabular-nums">
            Evaluated Hardware: {hardware.model} ({hardware.cores}C/{hardware.threads}T)
          </span>
        </div>
      </div>

      {/* 4-Quadrant Corpus Matrix */}
      <div className={styles.datasetMatrixStage}>
        {/* Quadrant 1: SIFT-100K (Active Testbed) */}
        <motion.div
          {...getEntranceProps(0)}
          className={`${styles.datasetQuad} ${styles.datasetQuadActive}`}
        >
          <div className={styles.datasetQuadHeader}>
            <div className="flex items-center gap-2">
              <span className={styles.datasetQuadTitle}>SIFT-100K Benchmark Testbed</span>
            </div>
            <span className="badge-pill emerald">
              <CheckCircle2 size={11} />
              <span>Active • Fully Evaluated</span>
            </span>
          </div>

          <p className="text-xs text-secondary">
            Standard computer-vision local image feature descriptors. Primary reference corpus for all empirical results reported in the paper.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Base Vectors / Queries:</span>
              <span className={styles.specVal}>100,000 / 10,000</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Dimensionality:</span>
              <span className={styles.specVal}>128-D FP32</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Metric Space:</span>
              <span className={styles.specVal}>Euclidean (L2)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Intrinsic Dimension:</span>
              <span className={`${styles.specVal} text-accent-glow`}>~9.8 (MLE Mean)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Peak Throughput:</span>
              <span className={`${styles.specVal} text-accent-glow`}>7,075.3 QPS (Step 5)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>RAM Footprint:</span>
              <span className={`${styles.specVal} text-emerald`}>22.54 MB (Step 6 SQ8)</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Ablation Protocol: Steps 1–6 complete &bull; Hardware: {hardware.model} (AVX2/FMA)
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

        {/* Quadrant 3: DBpedia-100K (NON-NEGOTIABLE: MUST REMAIN NOT RUN) */}
        <motion.div
          {...getEntranceProps(2)}
          className={`${styles.datasetQuad} ${styles.datasetQuadUnrun}`}
        >
          <div className={styles.datasetQuadHeader}>
            <span className={styles.datasetQuadTitle}>DBpedia-100K (OpenAI text-embedding-3-small)</span>
            <span className="badge-pill amber">
              <AlertTriangle size={11} />
              <span>NOT RUN &bull; PENDING EVALUATION</span>
            </span>
          </div>

          <p className="text-xs text-muted">
            Dense semantic text embeddings extracted from English Wikipedia abstracts. High intrinsic dimensionality and ambient noise regime.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Base Vectors:</span>
              <span className={styles.specVal}>100,000</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Embedding Dim:</span>
              <span className={styles.specVal}>768-D FP32</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Metric Space:</span>
              <span className={styles.specVal}>Angular / Cosine</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Estimated Index RAM:</span>
              <span className={styles.specVal}>~320 MB FP32</span>
            </div>
          </div>

          <div className={styles.unrunScientificCallout}>
            <strong>SCIENTIFIC INTEGRITY NOTICE:</strong>
            <span>Zero synthetic data. DBpedia-100K has not been evaluated on this hardware testbed. No metrics or projections are displayed.</span>
          </div>
        </motion.div>

        {/* Quadrant 4: GloVe-100 */}
        <motion.div
          {...getEntranceProps(3)}
          className={`${styles.datasetQuad} ${styles.datasetQuadUnrun}`}
        >
          <div className={styles.datasetQuadHeader}>
            <span className={styles.datasetQuadTitle}>GloVe-100 (Twitter Word Embeddings)</span>
            <span className="badge-pill">
              <Clock size={11} />
              <span>Queued for Sweep</span>
            </span>
          </div>

          <p className="text-xs text-muted">
            Global Vectors for Word Representation. Unnormalized token space with high hubness centrality around frequent semantic stop-words.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Base Vectors:</span>
              <span className={styles.specVal}>1,183,514</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Embedding Dim:</span>
              <span className={styles.specVal}>100-D FP32</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Metric Space:</span>
              <span className={styles.specVal}>Angular / Cosine</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Estimated Index RAM:</span>
              <span className={styles.specVal}>~650 MB FP32</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Queued for extended multi-million vector scalability benchmark sweep.
          </div>
        </motion.div>
      </div>
    </div>
  );
};
