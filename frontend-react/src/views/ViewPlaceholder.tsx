import React from 'react';
import type { ViewType } from '../types/benchmark';
import styles from './ViewPlaceholder.module.css';

interface Props {
  viewId: ViewType;
  title: string;
  isDataCritical?: boolean;
}

export const ViewPlaceholder: React.FC<Props> = ({ viewId, title, isDataCritical }) => {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.pretitle}>
          <span className="badge-pill">{viewId}</span>
          {isDataCritical && <span className="badge-pill accent">Data-Critical Phase</span>}
        </div>
        <h2 className={styles.title}>{title}</h2>
        <p className={styles.desc}>
          This view is scheduled in the sequential build order. Data-critical views (Overview, Benchmarks, Experiments, Datasets) are prioritized first.
        </p>
      </div>
    </div>
  );
};
