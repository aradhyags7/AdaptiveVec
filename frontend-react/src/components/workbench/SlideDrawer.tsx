import React from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import { X, Download, FileText } from 'lucide-react';
import styles from './SlideDrawer.module.css';

export const SlideDrawer: React.FC = () => {
  const { activeDrawer, closeDrawer, results, hardware } = useBenchmark();
  const isOpen = activeDrawer !== null;

  const renderContent = () => {
    switch (activeDrawer) {
      case 'data-export':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.sectionNotice}>
              Verified empirical benchmark data sourced directly from <span className="tabular-nums">benchmark_results.json</span>.
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
                <Download size={14} />
                <span>Export JSON Payload</span>
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
                <FileText size={14} />
                <span>Export CSV Matrix</span>
              </button>
            </div>

            <div className={styles.snippetBlock}>
              <div className={styles.snippetTitle}>LaTeX Table Generator Snippet</div>
              <pre className={styles.codeSnippet}>
{`\\begin{table}[h]
\\centering
\\begin{tabular}{lrrrr}
\\toprule
Step Configuration & Recall@10 & QPS & Edges & RAM (MB) \\\\
\\midrule
1. Baseline HNSW & 0.9913 & 4,708.1 & 2,709,125 & 59.93 \\\\
4. + Hubness Reg & 0.9854 & 5,452.0 & 2,510,334 & 59.17 \\\\
5. + Ada-ef Exit & 0.9745 & 7,075.3 & 2,509,138 & 59.16 \\\\
6. + INT8 SQ8    & 0.9594 & 2,987.6 & 2,509,743 & 22.54 \\\\
\\bottomrule
\\end{tabular}
\\caption{Empirical SIFT-100K testbed benchmarks on ${hardware.model}.}
\\end{table}`}
              </pre>
            </div>
          </div>
        );

      case 'inspector':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.nodeHeader}>
              <span className={styles.nodeTitle}>Node #48,219 Properties</span>
              <span className="badge-pill accent">Selected element</span>
            </div>
            <div className={styles.propertyList}>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Local Intrinsic Dim (MLE):</span>
                <span className="tabular-nums font-medium text-accent">26.4</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>k-NN Density (k=20):</span>
                <span className="tabular-nums">0.041</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Allocated Degree Capacity:</span>
                <span className="tabular-nums text-emerald font-medium">M = 22 / 24</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>efConstruction Budget:</span>
                <span className="tabular-nums">192</span>
              </div>
              <div className={styles.propRow}>
                <span className={styles.propKey}>Hierarchy Level:</span>
                <span className="tabular-nums">Layer 0 (Ground plane)</span>
              </div>
            </div>
            <div className={styles.subNote}>
              High LID regions expand local capacity budget to prevent boundary disconnectivity.
            </div>
          </div>
        );

      case 'index-builder':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.sectionNotice}>
              Interactive Policy Configuration Drawer (TODO — not yet wired).
            </div>
            <p className={styles.placeholderText}>
              Allows parameter tuning for $M \in [8, 28]$ and $efC \in [35, 200]$ before triggering real background index generation.
            </p>
          </div>
        );

      case 'settings':
        return (
          <div className={styles.drawerSection}>
            <div className={styles.propRow}>
              <span className={styles.propKey}>Interface Theme:</span>
              <span className="badge-pill">Persisted via localStorage</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propKey}>API Host:</span>
              <span className="tabular-nums">http://127.0.0.1:8000</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propKey}>Audio Micro-Haptics:</span>
              <span className="text-muted">Web Audio Synthesizer</span>
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
      case 'index-builder': return 'Index Policy Builder';
      case 'settings': return 'Workbench Settings';
      default: return 'Inspector';
    }
  };

  return (
    <>
      {/* Backdrop overlay */}
      <div
        className={`${styles.backdrop} ${isOpen ? styles.backdropOpen : ''}`}
        onClick={closeDrawer}
        aria-hidden={!isOpen}
      />

      {/* Slide Drawer */}
      <aside
        className={`${styles.drawer} ${isOpen ? styles.drawerOpen : ''}`}
        aria-label="Workbench Drawer"
        aria-hidden={!isOpen}
      >
        <div className={styles.drawerHeader}>
          <h3 className={styles.drawerTitle}>{getDrawerTitle()}</h3>
          <button
            className={styles.closeBtn}
            onClick={closeDrawer}
            title="Close Drawer (Esc)"
            aria-label="Close"
          >
            <X size={15} />
            <kbd className={styles.kbd}>Esc</kbd>
          </button>
        </div>

        <div className={styles.drawerBody}>
          {renderContent()}
        </div>
      </aside>
    </>
  );
};
