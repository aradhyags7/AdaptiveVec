import React from 'react';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  deltaText: string;
  deltaType?: 'positive' | 'negative' | 'neutral';
  baselineText: string;
  subtext?: string;
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  deltaText,
  deltaType = 'positive',
  baselineText,
  subtext,
  highlight = false,
}) => {
  const getBadgeClass = () => {
    if (deltaType === 'positive') return 'badge-pill emerald';
    if (deltaType === 'negative') return 'badge-pill amber';
    return 'badge-pill';
  };

  const getIcon = () => {
    if (deltaType === 'positive') return <ArrowUpRight size={11} />;
    if (deltaType === 'negative') return <ArrowDownRight size={11} />;
    return <Minus size={11} />;
  };

  return (
    <div className={`instrument-card ${highlight ? 'highlight' : ''}`} style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
    }}>
      {/* Top Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span className="stat-label">{label}</span>
          <span className={getBadgeClass()}>
            {getIcon()}
            <span>{deltaText}</span>
          </span>
        </div>

        {/* Value */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, margin: '4px 0 8px' }}>
          <span className="stat-number tabular-nums" style={{
            color: highlight ? 'var(--accent-glow)' : 'var(--text-primary)',
          }}>
            {value}
          </span>
          {unit && <span style={{ fontSize: '11.5px', color: 'var(--text-muted)', fontWeight: 600 }}>{unit}</span>}
        </div>
      </div>

      {/* Baseline Context & Subtext */}
      <div style={{
        borderTop: '1px solid var(--border-default)',
        paddingTop: 8,
        marginTop: 4,
        fontSize: '11px',
      }}>
        <div style={{ color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Baseline Reference:</span>
          <span className="tabular-nums" style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: '11.5px' }}>{baselineText}</span>
        </div>
        {subtext && (
          <div style={{ color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.45, fontWeight: 500 }}>
            {subtext}
          </div>
        )}
      </div>
    </div>
  );
};
