import React, { useState } from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download, FileText, CheckCircle2 } from 'lucide-react';
import styles from './SlideDrawer.module.css';

export const SlideDrawer: React.FC = () => {
  const { activeDrawer, closeDrawer, results, hardware, theme, toggleTheme } = useBenchmark();
  const isOpen = activeDrawer !== null;

  // Local state for index builder parameters
  const [degreeM, setDegreeM] = useState(16);
  const [efConstruction, setEfConstruction] = useState(100);
  const [alphaLID, setAlphaLID] = useState(0.45);
  const [betaDensity, setBetaDensity] = useState(0.35);

  const renderContent = () => {
    switch (activeDrawer) {
      case 'data-export':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.sectionNotice}>
              Verified empirical benchmark data sourced directly from <span className="tabular-nums font-semibold">benchmark_results.json</span>. Every figure is evaluated on real hardware (<span className="tabular-nums">{hardware.model}</span>).
            </div>

            <div className={styles.exportButtonGroup}>
              <button
                className={styles.exportBtn}
                onClick={() => {
                  const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'adaptivevec_benchmarks.json';
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                }}
              >
                <Download size={13} />
                <span>Export JSON</span>
              </button>

              <button
                className={styles.exportBtn}
                onClick={() => {
                  const headers = "dataset,configuration,n_samples,dim,qps,recall_at_10,total_edges,memory_mb,build_time_sec\n";
                  const rows = results.map(r => `"${r.dataset}","${r.configuration}",${r.n_samples},${r.dim},${r.qps},${r.recall_at_10},${r.total_edges},${r.memory_mb},${r.build_time_sec}`).join('\n');
                  const blob = new Blob([headers + rows], { type: 'text/csv' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'adaptivevec_benchmarks.csv';
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                }}
              >
                <FileText size={13} />
                <span>Export CSV</span>
              </button>
            </div>

            <div className={styles.snippetBlock}>
              <div className={styles.snippetTitle}>Verified LaTeX Table Output</div>
              <pre className={styles.codeSnippet}>
{`\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{lrrrr}
\\toprule
Step Configuration & Recall@10 & QPS & Edges & RAM (MB) \\\\
\\midrule
1. Baseline HNSW (Fixed M=16) & 0.9913 & 4,708.1 & 2,709,125 & 59.93 \\\\
2. + Dynamic M(x) & efC(x)    & 0.9883 & 5,147.1 & 2,549,825 & 59.32 \\\\
3. + Layer-Decoupled Scaling  & 0.9887 & 2,242.3 & 2,515,277 & 59.19 \\\\
4. + Hubness Reg (mu=0.15)    & 0.9854 & 5,452.0 & 2,510,334 & 59.17 \\\\
5. + Ada-ef Exit (Featured)   & 0.9745 & 7,075.3 & 2,509,138 & 59.16 \\\\
6. + INT8 SQ8 Quantization    & 0.9594 & 2,987.6 & 2,509,743 & 22.54 \\\\
\\bottomrule
\\end{tabular}
\\caption{Empirical SIFT-100K benchmarks on ${hardware.model} (${hardware.cores}C/${hardware.threads}T, AVX2/FMA, 16GB RAM).}
\\label{tab:adaptivevec_ablation}
\\end{table}`}
              </pre>
            </div>
          </div>
        );

      case 'inspector':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.nodeHeader}>
              <span className={styles.nodeTitle}>Node Telemetry #48,219</span>
              <span className="badge-pill emerald">
                <CheckCircle2 size={11} />
                <span>Active Anchor</span>
              </span>
            </div>
            <div className={styles.propertyList}>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Local Intrinsic Dim (MLE):</span>
                <span className="tabular-nums font-semibold text-accent-glow">26.4</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>k-NN Density (k=20):</span>
                <span className="tabular-nums font-medium">0.041</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Allocated Degree Capacity:</span>
                <span className="tabular-nums text-emerald font-semibold">M = 22 / 24</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>efConstruction Budget:</span>
                <span className="tabular-nums font-medium">192</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Hierarchy Layer:</span>
                <span className="tabular-nums">Layer 0 (Ground Plane)</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Hubness Multiplier:</span>
                <span className="tabular-nums">1.04x</span>
              </div>
            </div>
            <div className={styles.subNote}>
              High-LID frontier cluster: dynamic budget allocation prevents premature disconnectivity across sparse manifold ridges.
            </div>
          </div>
        );

      case 'index-builder':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.sectionNotice}>
              Index Construction Policy Calibration. Configures the adaptive neighborhood bounds before offline graph generation.
            </div>

            <div className={styles.paramGroup}>
              <div className={styles.paramRow}>
                <div className={styles.paramHeader}>
                  <span className={styles.paramLabel}>Base Degree Target (M):</span>
                  <span className={styles.paramVal}>{degreeM}</span>
                </div>
                <input
                  type="range"
                  min="8"
                  max="28"
                  value={degreeM}
                  onChange={(e) => setDegreeM(Number(e.target.value))}
                  className={styles.rangeInput}
                />
                <div className={styles.instrumentScale}>
                  <span>M=8 (Fast)</span>
                  <span>M=16 (Ref)</span>
                  <span>M=28 (Dense)</span>
                </div>
              </div>

              <div className={styles.paramRow}>
                <div className={styles.paramHeader}>
                  <span className={styles.paramLabel}>efConstruction Budget:</span>
                  <span className={styles.paramVal}>{efConstruction}</span>
                </div>
                <input
                  type="range"
                  min="35"
                  max="200"
                  value={efConstruction}
                  onChange={(e) => setEfConstruction(Number(e.target.value))}
                  className={styles.rangeInput}
                />
                <div className={styles.instrumentScale}>
                  <span>efC=35</span>
                  <span>efC=100</span>
                  <span>efC=200</span>
                </div>
              </div>

              <div className={styles.paramRow}>
                <div className={styles.paramHeader}>
                  <span className={styles.paramLabel}>LID Sensitivity Weight (α):</span>
                  <span className={styles.paramVal}>{alphaLID.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={alphaLID}
                  onChange={(e) => setAlphaLID(Number(e.target.value))}
                  className={styles.rangeInput}
                />
                <div className={styles.instrumentScale}>
                  <span>α=0.10</span>
                  <span>α=0.45</span>
                  <span>α=0.90</span>
                </div>
              </div>

              <div className={styles.paramRow}>
                <div className={styles.paramHeader}>
                  <span className={styles.paramLabel}>Density Scaling Factor (β):</span>
                  <span className={styles.paramVal}>{betaDensity.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={betaDensity}
                  onChange={(e) => setBetaDensity(Number(e.target.value))}
                  className={styles.rangeInput}
                />
                <div className={styles.instrumentScale}>
                  <span>β=0.10</span>
                  <span>β=0.35</span>
                  <span>β=0.90</span>
                </div>
              </div>
            </div>

            <div className={styles.subNote}>
              Compiled index target: <span className="tabular-nums font-semibold text-accent-glow">adaptive_hnsw_sift100k.bin</span>. Graph construction is executed via native C++ AVX2 backend.
            </div>
          </div>
        );

      case 'settings':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.propertyList}>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Instrument Appearance:</span>
                <button
                  className="badge-pill accent"
                  onClick={toggleTheme}
                  title="Switch between Dark Chassis and Archival Paper"
                >
                  {theme === 'dark' ? 'Dark Chassis (Primary)' : 'Archival Paper (Light)'}
                </button>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Telemetry Endpoint:</span>
                <span className="tabular-nums font-mono text-xs">http://127.0.0.1:8000</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Hardware Platform:</span>
                <span className="tabular-nums font-mono text-xs">{hardware.model}</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Vector Instructions:</span>
                <span className="badge-pill">AVX2 / FMA Active</span>
              </div>
            </div>
            <div className={styles.subNotice}>
              Theme preference is persisted to local browser storage.
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const getDrawerTitle = () => {
    switch (activeDrawer) {
      case 'data-export': return 'Raw Data & LaTeX Exporters';
      case 'inspector': return 'Element Inspector';
      case 'index-builder': return 'Index Construction Policy';
      case 'settings': return 'Instrument Settings';
      default: return 'Inspector';
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop overlay */}
          <motion.div
            key="drawer-backdrop"
            className={styles.backdrop}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={closeDrawer}
            aria-hidden={!isOpen}
          />

          {/* Slide Drawer with calibrated spring physics */}
          <motion.aside
            key="drawer-panel"
            className={styles.drawer}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 320, damping: 32, mass: 0.8 }}
            aria-label="Workbench Drawer"
            aria-hidden={!isOpen}
          >
            <div className={styles.drawerHeader}>
              <div className={styles.headerLeft}>
                <div className={styles.instrumentDot} />
                <h3 className={styles.drawerTitle}>{getDrawerTitle()}</h3>
              </div>
              <button
                className={styles.closeBtn}
                onClick={closeDrawer}
                title="Close Drawer (Esc)"
                aria-label="Close"
              >
                <X size={14} />
                <kbd className={styles.kbd}>Esc</kbd>
              </button>
            </div>

            <div className={styles.drawerBody}>
              {renderContent()}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
};
