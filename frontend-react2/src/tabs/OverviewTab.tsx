import React, { useState } from 'react';
import { MetricCard } from '../components/MetricCard';
import { MathView } from '../components/MathView';
import { PIPELINE_STAGES } from '../data/benchmarkData';
import { ArrowRight, CheckCircle2, ChevronRight } from 'lucide-react';

interface OverviewTabProps {
  onNavigateToBenchmarks: () => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({ onNavigateToBenchmarks }) => {
  const [selectedStage, setSelectedStage] = useState(0);

  const activeStageData = PIPELINE_STAGES[selectedStage];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Precision Workspace Bar / Hero Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        gap: 14,
        paddingBottom: 4,
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className="badge-pill accent">System Observatory</span>
            <span className="badge-pill">SIFT-100K Ground Truth Reference</span>
          </div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
            Adaptive Proximity Graph Indexing Engine
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '12px', maxWidth: 840, marginTop: 2 }}>
            Mitigating the curse of dimensionality and topological hubness in approximate nearest neighbor (ANN) search via pre-indexing local manifold probing, degree scaling, hubness regulation, and distance-stagnation search early exit.
          </p>
        </div>

        <button
          onClick={onNavigateToBenchmarks}
          className="btn-instrument primary"
          style={{ padding: '8px 16px', fontSize: '12px' }}
        >
          <span>Explore Live Benchmarks</span>
          <ArrowRight size={13} />
        </button>
      </div>

      {/* 4 Primary Key Performance Indicators */}
      <div className="grid-4">
        <MetricCard
          label="Search Throughput (QPS)"
          value="7,075.3"
          unit="QPS"
          deltaText="+50.3% Speedup"
          deltaType="positive"
          baselineText="4,708.1 QPS"
          subtext="Regime A (p=6 stagnation exit, ε=10⁻⁴)"
          highlight={true}
        />

        <MetricCard
          label="Memory Footprint (RAM)"
          value="22.5"
          unit="MB"
          deltaText="-62.4% Reduction"
          deltaType="positive"
          baselineText="59.9 MB"
          subtext="Regime B (8-bit asymmetric scalar quantization SQ8)"
          highlight={false}
        />

        <MetricCard
          label="Graph Construction Time"
          value="33.5"
          unit="sec"
          deltaText="-28.1% Faster"
          deltaType="positive"
          baselineText="46.6 sec"
          subtext="Layer-decoupled scaling + 7.4% fewer total graph edges"
        />

        <MetricCard
          label="Graph In-Degree Variance"
          value="64.4"
          unit="var"
          deltaText="-54.9% Hubness Drop"
          deltaType="positive"
          baselineText="142.8 var"
          subtext="Hubness penalty μ=0.15 eliminates search bottlenecks"
        />
      </div>

      {/* Interactive 5-Stage Algorithmic Pipeline */}
      <div className="instrument-card" style={{ padding: '18px 22px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="badge-pill accent">Architecture</span>
              <h3 style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                5-Stage AdaptiveVec Pipeline Architecture
              </h3>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>
              Inspect the mathematical formulation, problem solved, and measured impact across each pipeline stage.
            </p>
          </div>
          <span className="tabular-nums" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Stage {selectedStage + 1} of 5 Active
          </span>
        </div>

        {/* Stepper Tabs */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: 8,
          marginBottom: 16,
        }}>
          {PIPELINE_STAGES.map((stg, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedStage(idx)}
              style={{
                background: selectedStage === idx ? 'var(--bg-active)' : 'var(--bg-card-subtle)',
                border: selectedStage === idx ? '1px solid rgba(201, 125, 74, 0.5)' : '1px solid var(--border-default)',
                borderRadius: 'var(--radius-xs)',
                padding: '10px 12px',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.12s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="tabular-nums" style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  color: selectedStage === idx ? 'var(--accent-glow)' : 'var(--text-muted)',
                }}>
                  {stg.number}
                </span>
                {selectedStage === idx && <ChevronRight size={12} color="var(--accent-glow)" />}
              </div>
              <div style={{
                fontSize: '12px',
                fontWeight: 600,
                color: selectedStage === idx ? 'var(--text-primary)' : 'var(--text-secondary)',
                marginTop: 4,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {stg.name.split(' ')[1] || stg.name}
              </div>
            </button>
          ))}
        </div>

        {/* Detailed Stage Deep Dive Card */}
        <div style={{
          background: 'var(--bg-surface-sunken)',
          border: '1px solid var(--border-emphasis)',
          borderRadius: 'var(--radius-sm)',
          padding: 20,
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}>
          {/* Stage Header & Context */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <span className="badge-pill accent">Stage {activeStageData.number}</span>
                <h4 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {activeStageData.name}
                </h4>
                <span className="badge-pill illustrative">{activeStageData.tag}</span>
              </div>
              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: 850 }}>
                {activeStageData.summary}
              </p>
            </div>

            <div style={{
              padding: '8px 14px',
              borderRadius: 'var(--radius-xs)',
              background: 'var(--semantic-emerald-bg)',
              border: '1px solid rgba(107, 163, 126, 0.4)',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <CheckCircle2 size={15} color="var(--semantic-emerald)" />
              <div style={{ fontSize: '12px', color: 'var(--semantic-emerald)', fontWeight: 600 }}>
                {activeStageData.benefit}
              </div>
            </div>
          </div>

          {/* Crystal-Clear KaTeX / MathView Component */}
          <MathView
            title={`Stage ${activeStageData.number}: Core Mathematical Formulation`}
            tag={activeStageData.tag}
            latex={activeStageData.latex}
            fallbackText={activeStageData.formula}
            variables={activeStageData.variables}
            significance={activeStageData.significance}
            accentColor="var(--accent-glow)"
          />
        </div>
      </div>

      {/* Comparison: Standard HNSW vs AdaptiveVec */}
      {/* Comparison: Standard HNSW vs AdaptiveVec */}
      <div className="grid-2">
        <div className="instrument-card" style={{ borderTop: '2px solid var(--semantic-rose)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span className="badge-pill amber" style={{ fontSize: '11px', padding: '3px 8px' }}>Standard HNSW (Status Quo)</span>
            <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>BASELINE ARCHITECTURE</span>
          </div>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10, fontSize: '12.5px' }}>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-secondary)', background: 'rgba(217, 101, 91, 0.06)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(217, 101, 91, 0.15)' }}>
              <span style={{ color: 'var(--semantic-rose)', fontWeight: 800, fontSize: '13px' }}>✕</span>
              <span><strong style={{ color: 'var(--text-primary)' }}>Uniform Degree Allocation:</strong> Forces identical M=16 on dense centroids and sparse outliers alike.</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-secondary)', background: 'rgba(217, 101, 91, 0.06)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(217, 101, 91, 0.15)' }}>
              <span style={{ color: 'var(--semantic-rose)', fontWeight: 800, fontSize: '13px' }}>✕</span>
              <span><strong style={{ color: 'var(--text-primary)' }}>Severe Hubness Traps:</strong> A tiny fraction of dense nodes absorb thousands of in-degrees, becoming search bottlenecks.</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-secondary)', background: 'rgba(217, 101, 91, 0.06)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(217, 101, 91, 0.15)' }}>
              <span style={{ color: 'var(--semantic-rose)', fontWeight: 800, fontSize: '13px' }}>✕</span>
              <span><strong style={{ color: 'var(--text-primary)' }}>Exhaustive Fixed efSearch:</strong> Wastes 30%+ distance calculations continuing greedy hops after finding the true local minimum.</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-secondary)', background: 'rgba(217, 101, 91, 0.06)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(217, 101, 91, 0.15)' }}>
              <span style={{ color: 'var(--semantic-rose)', fontWeight: 800, fontSize: '13px' }}>✕</span>
              <span><strong style={{ color: 'var(--text-primary)' }}>FP32 Memory Bloat:</strong> Stores all 128 dimensions in full 32-bit floats, consuming 59.9 MB on 100K vectors.</span>
            </li>
          </ul>
        </div>

        <div className="instrument-card highlight" style={{ borderTop: '2px solid var(--semantic-emerald)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span className="badge-pill emerald" style={{ fontSize: '11px', padding: '3px 8px' }}>AdaptiveVec Innovations</span>
            <span style={{ fontSize: '10.5px', color: 'var(--semantic-emerald)', fontFamily: 'var(--font-mono)' }}>OPTIMIZED SOLUTION</span>
          </div>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10, fontSize: '12.5px' }}>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-primary)', background: 'rgba(107, 163, 126, 0.08)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(107, 163, 126, 0.25)' }}>
              <span style={{ color: 'var(--semantic-emerald)', fontWeight: 800, fontSize: '13px' }}>✓</span>
              <span><strong style={{ color: '#FFF' }}>LID-Guided Degree Scaling:</strong> Trims redundant edges from dense clusters, expanding M only on complex ridges (-7.4% total edges).</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-primary)', background: 'rgba(107, 163, 126, 0.08)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(107, 163, 126, 0.25)' }}>
              <span style={{ color: 'var(--semantic-emerald)', fontWeight: 800, fontSize: '13px' }}>✓</span>
              <span><strong style={{ color: '#FFF' }}>In-Degree Hubness Regulation:</strong> Slashes in-degree variance by 54.9% using soft penalty factor μ=0.15.</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-primary)', background: 'rgba(107, 163, 126, 0.08)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(107, 163, 126, 0.25)' }}>
              <span style={{ color: 'var(--semantic-emerald)', fontWeight: 800, fontSize: '13px' }}>✓</span>
              <span><strong style={{ color: '#FFF' }}>Distance Stagnation Early Exit:</strong> Safely cuts search hops at p=6 stagnation, achieving +50.3% QPS (7,075 QPS).</span>
            </li>
            <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, color: 'var(--text-primary)', background: 'rgba(107, 163, 126, 0.08)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(107, 163, 126, 0.25)' }}>
              <span style={{ color: 'var(--semantic-emerald)', fontWeight: 800, fontSize: '13px' }}>✓</span>
              <span><strong style={{ color: '#FFF' }}>Asymmetric SQ8 Quantization:</strong> Cuts RAM by 62.4% (down to 22.5 MB) with 95.9% empirical recall retention.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
