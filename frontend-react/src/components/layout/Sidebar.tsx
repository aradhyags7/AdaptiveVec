import React from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import type { ViewType } from '../../types/benchmark';
import {
  LayoutDashboard,
  Database,
  Hammer,
  Network,
  Search,
  LineChart,
  FlaskConical,
  Activity,
  Cpu,
  Settings,
  Zap
} from 'lucide-react';
import styles from './Sidebar.module.css';

interface NavItem {
  id: ViewType;
  label: string;
  icon: React.ReactNode;
  badge?: string;
  badgeType?: 'default' | 'demo' | 'accent';
}

const navItems: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={16} /> },
  { id: 'datasets', label: 'Datasets', icon: <Database size={16} /> },
  { id: 'index-builder', label: 'Index Builder', icon: <Hammer size={16} />, badge: 'Demo', badgeType: 'demo' },
  { id: 'graph-explorer', label: 'Graph Explorer', icon: <Network size={16} />, badge: 'Demo', badgeType: 'demo' },
  { id: 'query-lab', label: 'Query Lab', icon: <Search size={16} />, badge: 'Demo', badgeType: 'demo' },
  { id: 'benchmarks', label: 'Benchmarks', icon: <LineChart size={16} /> },
  { id: 'experiments', label: 'Experiments', icon: <FlaskConical size={16} /> },
  { id: 'node-analysis', label: 'Node Analysis', icon: <Activity size={16} />, badge: 'Demo', badgeType: 'demo' },
  { id: 'system-metrics', label: 'System Metrics', icon: <Cpu size={16} /> },
  { id: 'settings', label: 'Settings', icon: <Settings size={16} /> }
];

export const Sidebar: React.FC = () => {
  const { activeView, setActiveView } = useBenchmark();

  return (
    <aside className={styles.sidebar}>
      {/* Brand Header */}
      <div className={styles.brandContainer}>
        <div className={styles.logoRow}>
          <div className={styles.brandIcon}>
            <Zap size={14} className={styles.zapIcon} />
          </div>
          <span className={styles.brandTitle}>AdaptiveVec</span>
        </div>
        <div className={styles.researchTag}>Experimental research v0.4.2-alpha</div>
      </div>

      {/* Navigation Index */}
      <nav className={styles.nav}>
        <div className={styles.navSectionTitle}>Research index</div>
        <div className={styles.navList}>
          {navItems.map(item => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                className={`${styles.navItem} ${isActive ? styles.navItemActive : ''}`}
                onClick={() => setActiveView(item.id)}
              >
                <span className={styles.navIcon}>{item.icon}</span>
                <span className={styles.navLabel}>{item.label}</span>
                {item.badge && (
                  <span className={`${styles.navBadge} ${item.badgeType === 'demo' ? styles.badgeDemo : ''}`}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Sidebar Footer Metadata */}
      <div className={styles.sidebarFooter}>
        <div className={styles.footerRow}>
          <span className={styles.footerKey}>Commit</span>
          <span className={styles.footerVal}>git:5442aaf</span>
        </div>
        <div className={styles.footerRow}>
          <span className={styles.footerKey}>Instruction set</span>
          <span className={styles.footerVal}>AVX2 / FMA / FP32</span>
        </div>
        <div className={styles.footerRow}>
          <span className={styles.footerKey}>Resident RAM</span>
          <span className={styles.footerVal}>1.82 GB / 16.0 GB</span>
        </div>
      </div>
    </aside>
  );
};
