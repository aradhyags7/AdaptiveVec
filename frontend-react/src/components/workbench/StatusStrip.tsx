import React from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import styles from './StatusStrip.module.css';

export const StatusStrip: React.FC = () => {
  const { serverStatus, hardware } = useBenchmark();

  return (
    <footer className={styles.statusStrip} aria-label="Workbench Status Strip">
      {/* Left Section: Server Connection & Active Dataset */}
      <div className={styles.stripLeft}>
        <div className={styles.item}>
          <span className={`${styles.dot} ${serverStatus === 'online' ? styles.dotOnline : styles.dotOffline}`} />
          <span>{serverStatus === 'online' ? '127.0.0.1:8000' : 'Offline (Static)'}</span>
        </div>
        <span className={styles.sep}>|</span>
        <div className={styles.item}>
          <span className="tabular-nums">SIFT-100K (128-D FP32)</span>
        </div>
      </div>

      {/* Center Section: Core Benchmark Target */}
      <div className={styles.stripCenter}>
        <span className={styles.targetLabel}>Active Target:</span>{' '}
        <span className="tabular-nums text-accent font-medium">
          Step 5 Ada-ef: 7,075.3 QPS (0.9745 Recall@10 cf. 0.9856 at Step 4)
        </span>
      </div>

      {/* Right Section: Hardware Testbed & Version */}
      <div className={styles.stripRight}>
        <div className={styles.item}>
          <span>{hardware.model} ({hardware.instructionSet.split(' / ')[0]})</span>
          <span className={styles.sep}>•</span>
          <span className="tabular-nums">{hardware.ramGb}GB RAM</span>
        </div>
        <span className={styles.sep}>|</span>
        <div className={`${styles.item} tabular-nums`}>
          git:ee4339f
        </div>
      </div>
    </footer>
  );
};
