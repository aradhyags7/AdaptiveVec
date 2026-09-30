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
        {/* Quadrant 1: SIFT-100K & SIFT-250K (Active Testbed) */}
        <motion.div
          {...getEntranceProps(0)}
          className={`${styles.datasetQuad} ${styles.datasetQuadActive}`}
        >
          <div className={styles.datasetQuadHeader}>
            <div className="flex items-center gap-2">
              <span className={styles.datasetQuadTitle}>SIFT-100K & SIFT-250K Scale Expansion</span>
            </div>
            <span className="badge-pill emerald">
              <CheckCircle2 size={11} />
              <span>Active • Evaluated to 250K</span>
            </span>
          </div>

          <p className="text-xs text-secondary">
            Standard computer-vision local image feature descriptors (128-D FP32). Scaled 2.5× on local Intel Core 5 210H hardware to validate algorithm scaling properties.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Evaluated Scales:</span>
              <span className={styles.specVal}>100,000 & 250,000 Base Vectors</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Dimensionality:</span>
              <span className={styles.specVal}>128-D FP32 (Euclidean L2)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>250K Build Time:</span>
              <span className={`${styles.specVal} text-accent-glow`}>97.8 s (1.61× speedup vs 157.1 s)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>250K Graph Edges:</span>
              <span className={`${styles.specVal} text-emerald`}>6,154,914 (-8.55% / -575K links)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>250K Distance Hops:</span>
              <span className={`${styles.specVal} text-accent-glow`}>786.2 / query (-37.4% evals)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>250K Recall@10:</span>
              <span className={`${styles.specVal} text-emerald`}>0.9296 (Fast Early Exit)</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Empirical Scaling Sweep: 100K &rarr; 250K complete &bull; Hardware: {hardware.model} (12 Threads, AVX2/FMA)
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

        {/* Quadrant 4: SIFT-1M & Scalability Roadmap */}
        <motion.div
          {...getEntranceProps(3)}
          className={`${styles.datasetQuad} ${styles.datasetQuadUnrun}`}
        >
          <div className={styles.datasetQuadHeader}>
            <span className={styles.datasetQuadTitle}>SIFT-1M & GloVe Scalability Suite</span>
            <span className="badge-pill cyan">
              <Clock size={11} />
              <span>Binary Extracted &bull; Shift to Cluster</span>
            </span>
          </div>

          <p className="text-xs text-muted">
            1,000,000 base vectors pre-extracted to canonical binary (492 MB). Local PC evaluation achieved 1.91× speedup (241s vs 461s baseline); reserved for institutional server/cluster infrastructure transition for multi-million scaling.
          </p>

          <div className={styles.datasetSpecGrid}>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Target Dataset:</span>
              <span className={styles.specVal}>SIFT-1M & GloVe-100</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Base Vectors:</span>
              <span className={styles.specVal}>1,000,000 & 1,183,514</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Canonical Binary:</span>
              <span className={styles.specVal}>data/sift_base_1m.fvecs (492.1 MB)</span>
            </div>
            <div className={styles.datasetSpecItem}>
              <span className={styles.specKey}>Target Infrastructure:</span>
              <span className={`${styles.specVal} text-accent`}>Institutional Cluster / Cloud Server</span>
            </div>
          </div>

          <div className="text-xs text-muted font-mono" style={{ fontSize: '10px' }}>
            Infrastructure Roadmap: Reserved for multi-socket enterprise compute node to avoid local laptop thermal throttling.
          </div>
        </motion.div>
      </div>
    </div>
  );
};
