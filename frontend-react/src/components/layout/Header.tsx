import React from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import { Sun, Moon, Volume2, VolumeX, FileText, Search, ChevronDown } from 'lucide-react';
import styles from './Header.module.css';

export const Header: React.FC = () => {
  const { theme, toggleTheme, audioEnabled, setAudioEnabled, serverStatus, hardware } = useBenchmark();

  return (
    <header className={styles.header}>
      {/* Left Section: Active Dataset & Server Telemetry */}
      <div className={styles.headerLeft}>
        <div className={styles.datasetSelector}>
          <span className={styles.datasetDot} />
          <span className={styles.datasetName}>sift-100k-euclidean</span>
          <span className={styles.datasetParams}>(N=100,000, D=128, L2)</span>
          <ChevronDown size={12} className={styles.chevron} />
        </div>

        <div className={styles.telemetryPill}>
          <span className={`${styles.statusDot} ${serverStatus === 'online' ? styles.dotOnline : styles.dotOffline}`} />
          <span className={styles.statusText}>
            {serverStatus === 'online' ? 'Server online' : 'Static benchmark mode'}
          </span>
          <span className={styles.pillSep}>•</span>
          <span className={styles.hardwareText}>
            {hardware.model} ({hardware.instructionSet.split(' / ')[0]})
          </span>
        </div>
      </div>

      {/* Right Section: Actions & Settings */}
      <div className={styles.headerRight}>
        <button className={styles.searchButton} title="Quick Command / Search (Ctrl+K)">
          <Search size={13} />
          <span>Vector Search</span>
          <kbd className={styles.kbd}>Ctrl+K</kbd>
        </button>

        <a
          href="#paper"
          className={styles.paperLink}
          onClick={(e) => { e.preventDefault(); }}
          title="Research Specification"
        >
          <FileText size={13} />
          <span>arXiv:2403.xxxxx</span>
        </a>

        <button
          className={styles.iconButton}
          onClick={() => setAudioEnabled(!audioEnabled)}
          title={audioEnabled ? 'Disable Audio Feedback' : 'Enable Audio Feedback'}
          aria-label="Toggle Audio"
        >
          {audioEnabled ? <Volume2 size={15} /> : <VolumeX size={15} />}
        </button>

        <button
          className={styles.iconButton}
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          aria-label="Toggle Theme"
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </header>
  );
};
