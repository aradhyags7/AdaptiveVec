import React from 'react';
import { X, CheckCircle, HelpCircle, Shield, Award, Sparkles } from 'lucide-react';
import { PRESENTER_TALKING_POINTS } from '../data/benchmarkData';

interface PresenterGuideDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const PresenterGuideDrawer: React.FC<PresenterGuideDrawerProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      width: 'min(480px, 92vw)',
      height: '100vh',
      backgroundColor: 'var(--bg-sidebar)',
      borderLeft: '1px solid var(--border-emphasis)',
      boxShadow: 'var(--shadow-md)',
      zIndex: 100,
      display: 'flex',
      flexDirection: 'column',
      animation: 'slideIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
    }}>
      <style>{`
        @keyframes slideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>

      {/* Drawer Header */}
      <div style={{
        padding: '14px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-card)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            padding: 5,
            borderRadius: 'var(--radius-xs)',
            background: 'var(--accent-subtle)',
            border: '1px solid rgba(201, 125, 74, 0.35)',
          }}>
            <Award size={16} color="var(--accent-glow)" />
          </div>
          <div>
            <h2 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Project Evaluation Defense Notes
            </h2>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Quick talking points, technical answers, and empirical proof
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="btn-instrument"
          style={{ padding: '4px 8px' }}
          title="Close notes"
        >
          <X size={14} />
        </button>
      </div>

      {/* Drawer Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}>
        {/* Quick Pitch Box */}
        <div style={{
          background: 'var(--accent-subtle)',
          border: '1px solid rgba(201, 125, 74, 0.3)',
          borderRadius: 'var(--radius-sm)',
          padding: 14,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
            <Sparkles size={14} color="var(--accent-glow)" />
            <h3 style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-glow)' }}>
              The 30-Second Elevator Pitch
            </h3>
          </div>
          <p style={{ fontSize: '11.5px', color: 'var(--text-primary)', lineHeight: 1.6 }}>
            "Current state-of-the-art vector search engines like FAISS and Milvus use fixed, uniform graph connectivity for all vectors. But high-dimensional data is inherently inhomogeneous. <strong>AdaptiveVec</strong> is a topology-aware proximity graph engine that dynamically scales degree allocation by Local Intrinsic Dimensionality (LID), eliminates hubness bottlenecks via in-degree regulation, and terminates query search early using distance stagnation. On standard SIFT-100K benchmarks, this yields a <strong>+50.3% search throughput increase</strong> and up to <strong>62.4% RAM reduction</strong> while preserving 98.4% relative recall."
          </p>
        </div>

        {/* Structured Talking Points Sections */}
        {PRESENTER_TALKING_POINTS.map((sec, idx) => (
          <div key={idx} className="instrument-card" style={{ padding: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
              {idx === 0 && <Shield size={14} color="var(--semantic-amber)" />}
              {idx === 1 && <Sparkles size={14} color="var(--accent-glow)" />}
              {idx === 2 && <CheckCircle size={14} color="var(--semantic-emerald)" />}
              {idx === 3 && <HelpCircle size={14} color="var(--semantic-rose)" />}
              <h3 style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
                {sec.title}
              </h3>
            </div>

            <ul style={{ paddingLeft: 16, fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              {sec.bulletPoints.map((bp, bidx) => (
                <li key={bidx} style={{ marginBottom: 6 }}>
                  {bp.includes('Q:') ? (
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>{bp.split('→')[0]}</strong>
                      <div style={{ color: 'var(--accent-glow)', marginTop: 2 }}>
                        {bp.split('→')[1]}
                      </div>
                    </div>
                  ) : (
                    <span>{bp}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* Drawer Footer */}
      <div style={{
        padding: '10px 20px',
        borderTop: '1px solid var(--border-subtle)',
        background: 'var(--bg-card)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px',
        color: 'var(--text-muted)',
      }}>
        <span>Architecture: C++20 AVX2 + Python</span>
        <button
          onClick={onClose}
          className="btn-instrument"
          style={{ fontSize: '11px', padding: '4px 10px' }}
        >
          Close
        </button>
      </div>
    </div>
  );
};
