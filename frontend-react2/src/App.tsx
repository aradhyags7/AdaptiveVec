import React, { useState } from 'react';
import { HeaderBar } from './components/HeaderBar';
import { PresenterGuideDrawer } from './components/PresenterGuideDrawer';
import { OverviewTab } from './tabs/OverviewTab';
import { BenchmarksTab } from './tabs/BenchmarksTab';
import { QuerySimulatorTab } from './tabs/QuerySimulatorTab';
import { TheoryTab } from './tabs/TheoryTab';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [guideOpen, setGuideOpen] = useState<boolean>(false);

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab onNavigateToBenchmarks={() => setActiveTab('benchmarks')} />;
      case 'benchmarks':
        return <BenchmarksTab />;
      case 'simulator':
        return <QuerySimulatorTab />;
      case 'theory':
        return <TheoryTab />;
      default:
        return <OverviewTab onNavigateToBenchmarks={() => setActiveTab('benchmarks')} />;
    }
  };

  return (
    <div className="app-container">
      {/* Precision Header */}
      <HeaderBar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onToggleGuide={() => setGuideOpen(!guideOpen)}
        guideOpen={guideOpen}
      />

      {/* Main Dynamic View Area */}
      <main className="main-content">
        {renderContent()}
      </main>

      {/* Slide-out Presenter Defense Notes Drawer */}
      <PresenterGuideDrawer
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
      />

      {/* Professional Scientific Status Strip Footer */}
      <footer style={{
        height: 28,
        minHeight: 28,
        borderTop: '1px solid var(--border-subtle)',
        background: 'var(--bg-card-subtle)',
        padding: '0 24px',
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-muted)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        userSelect: 'none',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: 'var(--text-secondary)' }}>ADAPTIVEVEC INSTRUMENT PLATFORM</span>
          <span>/</span>
          <span>v2.4.0</span>
          <span>/</span>
          <span className="text-emerald">AVX2-SIMD x86_64 READY</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span>CORPUS: SIFT-100K (128-DIM)</span>
          <span>/</span>
          <span style={{ color: 'var(--accent-glow)', fontWeight: 600 }}>TOPOLOGY-AWARE PROXIMITY GRAPH</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
