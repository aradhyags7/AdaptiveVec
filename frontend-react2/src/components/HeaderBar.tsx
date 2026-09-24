import React from 'react';
import { Layers, Activity, Cpu, FileText, HelpCircle } from 'lucide-react';

interface HeaderBarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onToggleGuide: () => void;
  guideOpen: boolean;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  activeTab,
  setActiveTab,
  onToggleGuide,
  guideOpen,
}) => {
  return (
    <header style={{
      height: 48,
      minHeight: 48,
      padding: '0 24px',
      borderBottom: '1px solid var(--border-subtle)',
      background: 'var(--bg-card)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 40,
    }}>
      {/* Left: Brand Identity */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 28,
          height: 28,
          borderRadius: 'var(--radius-xs)',
          background: 'var(--accent-subtle)',
          border: '1px solid rgba(201, 125, 74, 0.4)',
        }}>
          <Layers size={16} color="var(--accent-glow)" />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontSize: '13.5px',
            fontWeight: 700,
            letterSpacing: '-0.01em',
            color: 'var(--text-primary)',
          }}>
            AdaptiveVec
          </span>
          <span className="badge-pill accent">v2.4 Engine</span>
          <span className="badge-pill emerald">
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--semantic-emerald)', display: 'inline-block' }} />
            AVX2 SIMD Active
          </span>
        </div>
      </div>

      {/* Center: Instrument Segmented Navigation */}
      <div className="segmentedGroup">
        <button
          className={`segBtn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <Activity size={13} />
          <span>Overview</span>
        </button>

        <button
          className={`segBtn ${activeTab === 'benchmarks' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmarks')}
        >
          <Layers size={13} />
          <span>Benchmarks & Pareto</span>
        </button>

        <button
          className={`segBtn ${activeTab === 'simulator' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulator')}
        >
          <Cpu size={13} />
          <span>Trajectory Simulator</span>
        </button>

        <button
          className={`segBtn ${activeTab === 'theory' ? 'active' : ''}`}
          onClick={() => setActiveTab('theory')}
        >
          <FileText size={13} />
          <span>Theory & Architecture</span>
        </button>
      </div>

      {/* Right Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          className={`btn-instrument ${guideOpen ? 'emerald' : 'primary'}`}
          onClick={onToggleGuide}
          title="Toggle presenter defense talking points and examiner QA"
        >
          <HelpCircle size={13} />
          <span>{guideOpen ? 'Close Defense Notes' : 'Evaluation Defense Guide'}</span>
        </button>
      </div>
    </header>
  );
};
