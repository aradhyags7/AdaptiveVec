import React, { useState } from 'react';
import {
  SIFT_ABLATION_STEPS,
  DATASETS_BENCHMARK,
  STAGNATION_WINDOW_SWEEP,
  HUBNESS_SWEEP,
  BenchmarkPoint,
} from '../data/benchmarkData';
import {
  TrendingUp,
  BarChart2,
  Table,
  Sliders,
  Image as ImageIcon,
  Maximize2,
  X,
} from 'lucide-react';

export const BenchmarksTab: React.FC = () => {
  const [subTab, setSubTab] = useState<'pareto' | 'ablation' | 'datasets' | 'figures' | 'playground'>('pareto');
  const [selectedStep, setSelectedStep] = useState<BenchmarkPoint>(SIFT_ABLATION_STEPS[4]); // Default to Step 5 (Regime A)
  const [selectedSweepP, setSelectedSweepP] = useState<number | string>(6);
  const [selectedMu, setSelectedMu] = useState<number>(0.15);
  const [zoomImage, setZoomImage] = useState<{ src: string; title: string; caption: string } | null>(null);

  // Active sweep points
  const currentSweepPoint = STAGNATION_WINDOW_SWEEP.find((s) => s.p === selectedSweepP) || STAGNATION_WINDOW_SWEEP[2];
  const currentMuPoint = HUBNESS_SWEEP.find((h) => h.mu === selectedMu) || HUBNESS_SWEEP[2];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Header & Sub-Navigation */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 12,
        paddingBottom: 2,
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className="badge-pill emerald">Empirical Validation Suite</span>
            <span className="badge-pill">Texmex SIFT-100K Ground Truth</span>
          </div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>
            Empirical Benchmark Observatory
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            Verified evaluation results across 10,000 queries on fixed random seeds under standard ANN-Benchmarks protocols.
          </p>
        </div>

        {/* Sub-Tabs Selector */}
        <div className="segmentedGroup">
          <button
            className={`segBtn ${subTab === 'pareto' ? 'active' : ''}`}
            onClick={() => setSubTab('pareto')}
          >
            <TrendingUp size={12} />
            <span>Pareto Frontier</span>
          </button>

          <button
            className={`segBtn ${subTab === 'ablation' ? 'active' : ''}`}
            onClick={() => setSubTab('ablation')}
          >
            <BarChart2 size={12} />
            <span>6-Step Ablation</span>
          </button>

          <button
            className={`segBtn ${subTab === 'datasets' ? 'active' : ''}`}
            onClick={() => setSubTab('datasets')}
          >
            <Table size={12} />
            <span>Cross-Dataset Matrix</span>
          </button>

          <button
            className={`segBtn ${subTab === 'figures' ? 'active' : ''}`}
            onClick={() => setSubTab('figures')}
          >
            <ImageIcon size={12} />
            <span>Publication Plots (4)</span>
          </button>

          <button
            className={`segBtn ${subTab === 'playground' ? 'active' : ''}`}
            onClick={() => setSubTab('playground')}
          >
            <Sliders size={12} />
            <span>Sensitivity Playground</span>
          </button>
        </div>
      </div>

      {/* SUB-VIEW 1: PARETO FRONTIER */}
      {subTab === 'pareto' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="instrument-card" style={{ padding: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div>
                <h3 style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  SIFT-100K Throughput vs. Recall Pareto Frontier
                </h3>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Interactive frontier mapping search throughput (QPS) against Recall@10 across stagnation thresholds p ∈ [2, 10, ∞].
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <span className="badge-pill accent">● AdaptiveVec Frontier</span>
                <span className="badge-pill amber">▲ Baseline HNSW</span>
              </div>
            </div>

            {/* SVG Interactive Pareto Curve */}
            <div style={{
              background: 'var(--canvas-bg)',
              borderRadius: 'var(--radius-xs)',
              padding: '20px 16px',
              border: '1px solid var(--border-default)',
              position: 'relative',
            }}>
              <svg viewBox="0 0 900 340" style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
                <defs>
                  <linearGradient id="amberCurveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#C97D4A" />
                    <stop offset="50%" stopColor="#E8A25C" />
                    <stop offset="100%" stopColor="#D9A74A" />
                  </linearGradient>
                  <filter id="amberGlow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>
                </defs>

                {/* Grid Lines */}
                {[0.92, 0.94, 0.96, 0.98, 1.0].map((val, idx) => {
                  const x = 120 + ((val - 0.91) / (1.0 - 0.91)) * 720;
                  return (
                    <g key={idx}>
                      <line x1={x} y1={30} x2={x} y2={280} stroke="rgba(225, 210, 190, 0.05)" strokeDasharray="3 3" />
                      <text x={x} y={300} fill="var(--text-dim)" fontSize="10.5" textAnchor="middle" fontFamily="var(--font-mono)">
                        {(val * 100).toFixed(0)}%
                      </text>
                    </g>
                  );
                })}

                {[4000, 5000, 6000, 7000, 8000].map((qpsVal, idx) => {
                  const y = 280 - ((qpsVal - 3500) / (9000 - 3500)) * 250;
                  return (
                    <g key={idx}>
                      <line x1={100} y1={y} x2={860} y2={y} stroke="rgba(225, 210, 190, 0.05)" strokeDasharray="3 3" />
                      <text x={90} y={y + 4} fill="var(--text-dim)" fontSize="10.5" textAnchor="end" fontFamily="var(--font-mono)">
                        {qpsVal.toLocaleString()}
                      </text>
                    </g>
                  );
                })}

                {/* Axes */}
                <line x1={100} y1={280} x2={860} y2={280} stroke="var(--border-default)" strokeWidth="1" />
                <line x1={100} y1={30} x2={100} y2={280} stroke="var(--border-default)" strokeWidth="1" />
                <text x={480} y={326} fill="var(--text-muted)" fontSize="11" textAnchor="middle" fontWeight="600" fontFamily="var(--font-sans)">
                  Recall@10 Accuracy
                </text>
                <text x={25} y={155} fill="var(--text-muted)" fontSize="11" textAnchor="middle" fontWeight="600" fontFamily="var(--font-sans)" transform="rotate(-90 25 155)">
                  Throughput (QPS)
                </text>

                {/* Pareto Spline Curve connecting the points */}
                <path
                  d="M 212 57 Q 380 92 636 118 T 729 122 T 768 147 T 763 191"
                  fill="none"
                  stroke="url(#amberCurveGrad)"
                  strokeWidth="3"
                  filter="url(#amberGlow)"
                />

                {/* Baseline HNSW point (Recall 0.9913, QPS 4,708) */}
                {(() => {
                  const bx = 120 + ((0.9913 - 0.91) / (1.0 - 0.91)) * 720;
                  const by = 280 - ((4708.1 - 3500) / (9000 - 3500)) * 250;
                  return (
                    <g>
                      <polygon
                        points={`${bx},${by - 8} ${bx + 8},${by + 6} ${bx - 8},${by + 6}`}
                        fill="var(--semantic-amber)"
                        stroke="#ffffff"
                        strokeWidth="1.2"
                      />
                      <text x={bx + 12} y={by + 4} fill="var(--semantic-amber)" fontSize="10.5" fontWeight="700" fontFamily="var(--font-sans)">
                        Baseline HNSW (4,708 QPS, 0.991)
                      </text>
                    </g>
                  );
                })()}

                {/* AdaptiveVec Sweep Points */}
                {STAGNATION_WINDOW_SWEEP.map((pt, idx) => {
                  const px = 120 + (((pt.recall as number) - 0.91) / (1.0 - 0.91)) * 720;
                  const py = 280 - ((pt.qps - 3500) / (9000 - 3500)) * 250;
                  const isSelected = selectedSweepP === pt.p;
                  const isRegimeA = pt.p === 6;

                  return (
                    <g
                      key={idx}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSelectedSweepP(pt.p)}
                    >
                      <circle
                        cx={px}
                        cy={py}
                        r={isSelected ? 9 : isRegimeA ? 7 : 5.5}
                        fill={isRegimeA ? 'var(--accent-glow)' : 'var(--accent)'}
                        stroke="#ffffff"
                        strokeWidth={isSelected ? 2.5 : 1.2}
                        filter={isSelected || isRegimeA ? 'url(#amberGlow)' : undefined}
                      />
                      <text
                        x={px}
                        y={py - 12}
                        fill={isRegimeA ? 'var(--accent-glow)' : 'var(--text-secondary)'}
                        fontSize="10.5"
                        fontWeight={isRegimeA || isSelected ? '700' : '500'}
                        textAnchor="middle"
                        fontFamily="var(--font-mono)"
                      >
                        {pt.p === '∞' ? 'p=∞' : `p=${pt.p}`}
                      </text>
                      {isRegimeA && (
                        <text x={px} y={py + 22} fill="var(--semantic-emerald)" fontSize="9.5" fontWeight="700" textAnchor="middle" fontFamily="var(--font-mono)">
                          ★ Regime A (+50.3%)
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Sweep Point Detailed Inspector */}
            <div style={{
              marginTop: 14,
              background: 'var(--bg-card-subtle)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-xs)',
              padding: 16,
              display: 'grid',
              gridTemplateColumns: 'repeat(5, 1fr)',
              gap: 14,
              alignItems: 'center',
            }}>
              <div>
                <div className="stat-label">Selected Threshold</div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-glow)', marginTop: 2 }}>
                  {currentSweepPoint.label}
                </div>
              </div>

              <div>
                <div className="stat-label">Throughput (QPS)</div>
                <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {currentSweepPoint.qps.toLocaleString()}
                </div>
                <span className="badge-pill emerald" style={{ marginTop: 3 }}>
                  +{currentSweepPoint.qpsGainPct}% vs Base
                </span>
              </div>

              <div>
                <div className="stat-label">Recall@10</div>
                <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {(currentSweepPoint.recall * 100).toFixed(2)}%
                </div>
                <span className="badge-pill" style={{ marginTop: 3 }}>
                  -{currentSweepPoint.recallDropPct}% delta
                </span>
              </div>

              <div>
                <div className="stat-label">Evals / Query</div>
                <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-secondary)', marginTop: 2 }}>
                  {currentSweepPoint.evals}
                </div>
                <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                  (Base: 1,121.2)
                </span>
              </div>

              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', lineHeight: 1.5, borderLeft: '1px solid var(--border-subtle)', paddingLeft: 14 }}>
                <strong style={{ color: 'var(--text-secondary)' }}>Operational Verdict:</strong>
                <div style={{ color: 'var(--text-primary)', marginTop: 2 }}>{currentSweepPoint.description}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-VIEW 2: 6-STEP ABLATION WATERFALL */}
      {subTab === 'ablation' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="instrument-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge-pill accent">Component Attribution</span>
                <h3 style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  6-Step Cumulative Architectural Ablation on SIFT-100K
                </h3>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>
                Each step isolates an algorithmic component to measure its individual contribution to throughput, memory, and graph efficiency.
              </p>
            </div>

            {/* Waterfall Horizontal Stacked Cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
              {SIFT_ABLATION_STEPS.map((step) => {
                const isSelected = selectedStep.id === step.id;
                const qpsPct = (step.qps / 7500) * 100;
                const ramPct = (step.memoryMb / 65) * 100;

                return (
                  <div
                    key={step.id}
                    onClick={() => setSelectedStep(step)}
                    style={{
                      background: isSelected ? 'var(--bg-active)' : 'var(--bg-card-subtle)',
                      border: isSelected ? '1px solid rgba(201, 125, 74, 0.45)' : '1px solid var(--border-default)',
                      borderRadius: 'var(--radius-xs)',
                      padding: '10px 14px',
                      cursor: 'pointer',
                      transition: 'all 0.12s ease',
                      display: 'grid',
                      gridTemplateColumns: '2fr 1.5fr 1.2fr 1.2fr 1.5fr',
                      alignItems: 'center',
                      gap: 14,
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontWeight: 600, color: isSelected ? 'var(--accent-glow)' : 'var(--text-primary)', fontSize: '12px' }}>
                          {step.name}
                        </span>
                        {step.highlight && <span className="badge-pill emerald">Featured</span>}
                        {step.regime === 'Regime B (Low RAM)' && <span className="badge-pill accent">Low RAM</span>}
                      </div>
                      <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: 2 }}>
                        {step.mechanism}
                      </div>
                    </div>

                    {/* QPS Bar */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', marginBottom: 3 }}>
                        <span style={{ color: 'var(--text-muted)' }}>Throughput</span>
                        <span className="tabular-nums" style={{ color: 'var(--accent-glow)', fontWeight: 600 }}>
                          {step.qps.toLocaleString()} QPS
                        </span>
                      </div>
                      <div style={{ width: '100%', height: 4, background: 'var(--bg-app)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ width: `${qpsPct}%`, height: '100%', background: 'var(--accent)', borderRadius: 2 }} />
                      </div>
                    </div>

                    {/* RAM Bar */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', marginBottom: 3 }}>
                        <span style={{ color: 'var(--text-muted)' }}>RAM</span>
                        <span className="tabular-nums" style={{ color: step.memoryMb < 30 ? 'var(--semantic-emerald)' : 'var(--text-secondary)', fontWeight: 600 }}>
                          {step.memoryMb} MB
                        </span>
                      </div>
                      <div style={{ width: '100%', height: 4, background: 'var(--bg-app)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ width: `${ramPct}%`, height: '100%', background: step.memoryMb < 30 ? 'var(--semantic-emerald)' : 'var(--border-emphasis)', borderRadius: 2 }} />
                      </div>
                    </div>

                    {/* Recall */}
                    <div>
                      <div style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>Recall@10</div>
                      <div className="tabular-nums" style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-primary)', marginTop: 1 }}>
                        {(step.recall * 100).toFixed(2)}%
                      </div>
                    </div>

                    {/* Delta Badge */}
                    <div style={{ textAlign: 'right' }}>
                      <span className="badge-pill" style={{ fontSize: '10.5px' }}>
                        {step.deltaText}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Selected Step Explanation Card */}
            <div style={{
              background: 'var(--bg-surface-sunken)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-xs)',
              padding: 16,
              display: 'grid',
              gridTemplateColumns: '1.5fr 1fr',
              gap: 16,
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <span className="badge-pill accent">Step {selectedStep.step} Deep Dive</span>
                  <h4 style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {selectedStep.name}
                  </h4>
                </div>
                <p style={{ fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                  {selectedStep.whyItMatters}
                </p>
              </div>

              <div style={{
                background: 'var(--bg-card-subtle)',
                borderRadius: 'var(--radius-xs)',
                padding: '10px 14px',
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 10,
                fontSize: '11px',
              }}>
                <div>
                  <div style={{ color: 'var(--text-muted)' }}>Total Graph Edges:</div>
                  <div className="tabular-nums" style={{ fontWeight: 600, color: 'var(--text-primary)', marginTop: 1 }}>
                    {selectedStep.edges.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)' }}>Build Time:</div>
                  <div className="tabular-nums" style={{ fontWeight: 600, color: 'var(--text-primary)', marginTop: 1 }}>
                    {selectedStep.buildSec} sec
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)' }}>Distance Evals/Query:</div>
                  <div className="tabular-nums" style={{ fontWeight: 600, color: 'var(--accent-glow)', marginTop: 1 }}>
                    {selectedStep.evals} evals
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)' }}>Throughput Delta:</div>
                  <div className="tabular-nums" style={{ fontWeight: 600, color: 'var(--semantic-emerald)', marginTop: 1 }}>
                    {((selectedStep.qps / 4708.1 - 1) * 100).toFixed(1)}% vs Base
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-VIEW 3: CROSS-DATASET MATRIX */}
      {subTab === 'datasets' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="instrument-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge-pill accent">Generalization Suite</span>
                <h3 style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Multi-Domain Cross-Dataset Evaluation Suite
                </h3>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>
                Consistent speedups across Computer Vision (SIFT), NLP Embeddings (GloVe), Image Features (Fashion-MNIST), and Synthetic Multi-Clusters.
              </p>
            </div>

            <div className="grid-2">
              {DATASETS_BENCHMARK.map((ds, idx) => (
                <div key={idx} className="instrument-card" style={{ padding: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                    <div>
                      <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
                        {ds.name}
                      </h4>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {ds.dim}-dim • {ds.vectors} vectors • {ds.domain}
                      </div>
                    </div>
                    <span className="badge-pill emerald">+{ds.qpsSpeedupPct}% QPS</span>
                  </div>

                  <table className="instrument-table" style={{ marginTop: 6 }}>
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Standard HNSW</th>
                        <th>AdaptiveVec</th>
                        <th>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Throughput (QPS)</td>
                        <td className="tabular-nums">{ds.baselineQps.toLocaleString()}</td>
                        <td className="tabular-nums" style={{ color: 'var(--accent-glow)', fontWeight: 600 }}>
                          {ds.adaptiveQps.toLocaleString()}
                        </td>
                        <td>
                          <span className="badge-pill emerald">+{ds.qpsSpeedupPct}%</span>
                        </td>
                      </tr>
                      <tr>
                        <td>Recall@10</td>
                        <td className="tabular-nums">{(ds.baselineRecall * 100).toFixed(1)}%</td>
                        <td className="tabular-nums" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                          {(ds.adaptiveRecall * 100).toFixed(1)}%
                        </td>
                        <td>
                          <span className="badge-pill">
                            -{((ds.baselineRecall - ds.adaptiveRecall) * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td>RAM Footprint</td>
                        <td className="tabular-nums">{ds.baselineRamMb} MB</td>
                        <td className="tabular-nums" style={{ color: 'var(--semantic-emerald)', fontWeight: 600 }}>
                          {ds.adaptiveRamMb} MB
                        </td>
                        <td>
                          <span className="badge-pill accent">-{ds.ramReductionPct}%</span>
                        </td>
                      </tr>
                      <tr>
                        <td>Build Time</td>
                        <td className="tabular-nums">{ds.baselineBuildSec}s</td>
                        <td className="tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                          {ds.adaptiveBuildSec}s
                        </td>
                        <td>
                          <span className="badge-pill">
                            -{((1 - ds.adaptiveBuildSec / ds.baselineBuildSec) * 100).toFixed(0)}%
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* SUB-VIEW 4: PUBLICATION PLOTS GALLERY */}
      {subTab === 'figures' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="instrument-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge-pill accent">Research Paper Visualizations</span>
                <h3 style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  High-DPI Figures Included in IEEE/ACM Research Paper
                </h3>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>
                Rendered from 300 DPI vector publication scripts. Click any plot to expand full screen.
              </p>
            </div>

            <div className="grid-2">
              {[
                {
                  src: '/paper_figures/fig1_pareto.png',
                  title: 'Figure 1: SIFT-100K Throughput-Recall Pareto Frontier',
                  caption: 'Throughput (QPS) vs. Recall@10 across varying early-exit stagnation windows p ∈ [2, 10, ∞] against baseline HNSW. Regime A achieves +50.3% QPS.',
                },
                {
                  src: '/paper_figures/fig2_hubness_distribution.png',
                  title: 'Figure 2: In-Degree Centrality Hubness Distribution',
                  caption: 'Probability density function of node in-degrees for Baseline HNSW vs. AdaptiveVec regulated under μ = 0.15, slashing variance by 54.9%.',
                },
                {
                  src: '/paper_figures/fig3_ablation_waterfall.png',
                  title: 'Figure 3: 6-Step Cumulative Architectural Ablation',
                  caption: 'Component-by-component waterfall breakdown showing QPS throughput gains (left axis) and RAM footprint reduction under SQ8 (right axis).',
                },
                {
                  src: '/paper_figures/fig4_synthetic_sweep.png',
                  title: 'Figure 4: Synthetic-Multi-Cluster Hubness Penalty Sweep',
                  caption: 'Sensitivity sweep of hubness regulation coefficient μ ∈ [0.0, 0.30] showing the trade-off between in-degree balance and routing fidelity.',
                },
              ].map((fig, idx) => (
                <div
                  key={idx}
                  className="instrument-card"
                  style={{
                    padding: 14,
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10,
                  }}
                  onClick={() => setZoomImage(fig)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {fig.title}
                    </span>
                    <button
                      className="btn-instrument"
                      style={{ padding: '2px 6px', fontSize: '10.5px' }}
                      title="Enlarge figure"
                    >
                      <Maximize2 size={11} />
                    </button>
                  </div>

                  <div style={{
                    width: '100%',
                    height: 220,
                    borderRadius: 'var(--radius-xs)',
                    overflow: 'hidden',
                    background: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <img
                      src={fig.src}
                      alt={fig.title}
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  </div>

                  <p style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                    {fig.caption}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* SUB-VIEW 5: PARAMETER SENSITIVITY PLAYGROUND */}
      {subTab === 'playground' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="instrument-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge-pill accent">Interactive Telemetry</span>
                <h3 style={{ fontSize: '13.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Hyperparameter Sensitivity & Physics Playground
                </h3>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>
                Adjust the early-exit stagnation threshold (p) and hubness penalty (μ) below to see their real-time effect on graph topology and search dynamics.
              </p>
            </div>

            <div className="grid-2">
              {/* Selector 1: Early Exit Stagnation Window (p) */}
              <div style={{ background: 'var(--bg-surface-sunken)', padding: 16, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-default)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <label className="stat-label">
                    Stagnation Window (p)
                  </label>
                  <span className="tabular-nums" style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-glow)' }}>
                    p = {selectedSweepP}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                  {[2, 4, 6, 8, 10, '∞'].map((val) => (
                    <button
                      key={String(val)}
                      onClick={() => setSelectedSweepP(val)}
                      style={{
                        flex: 1,
                        padding: '6px 0',
                        borderRadius: 'var(--radius-xs)',
                        border: selectedSweepP === val ? '1px solid var(--accent)' : '1px solid var(--border-default)',
                        background: selectedSweepP === val ? 'var(--accent-subtle)' : 'var(--bg-card)',
                        color: selectedSweepP === val ? 'var(--accent-glow)' : 'var(--text-muted)',
                        fontWeight: 600,
                        cursor: 'pointer',
                        fontSize: '11px',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      {val === '∞' ? '∞' : val}
                    </button>
                  ))}
                </div>

                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.5, background: 'var(--bg-card-subtle)', padding: 10, borderRadius: 'var(--radius-xs)' }}>
                  <strong style={{ color: 'var(--text-primary)' }}>Operational Effect:</strong>
                  <div style={{ marginTop: 2, color: 'var(--text-muted)' }}>
                    {currentSweepPoint.description}
                  </div>
                </div>
              </div>

              {/* Selector 2: Hubness Penalty (mu) */}
              <div style={{ background: 'var(--bg-surface-sunken)', padding: 16, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-default)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <label className="stat-label">
                    Hubness In-Degree Penalty (μ)
                  </label>
                  <span className="tabular-nums" style={{ fontSize: '13px', fontWeight: 700, color: 'var(--semantic-emerald)' }}>
                    μ = {selectedMu.toFixed(2)}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                  {[0.0, 0.05, 0.15, 0.30].map((mVal) => (
                    <button
                      key={mVal}
                      onClick={() => setSelectedMu(mVal)}
                      style={{
                        flex: 1,
                        padding: '6px 0',
                        borderRadius: 'var(--radius-xs)',
                        border: selectedMu === mVal ? '1px solid var(--semantic-emerald)' : '1px solid var(--border-default)',
                        background: selectedMu === mVal ? 'var(--semantic-emerald-bg)' : 'var(--bg-card)',
                        color: selectedMu === mVal ? 'var(--semantic-emerald)' : 'var(--text-muted)',
                        fontWeight: 600,
                        cursor: 'pointer',
                        fontSize: '11px',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      {mVal.toFixed(2)}
                    </button>
                  ))}
                </div>

                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.5, background: 'var(--bg-card-subtle)', padding: 10, borderRadius: 'var(--radius-xs)' }}>
                  <strong style={{ color: 'var(--text-primary)' }}>Graph In-Degree Physics:</strong>
                  <div style={{ marginTop: 2, color: 'var(--text-muted)' }}>
                    {currentMuPoint.verdict}
                  </div>
                </div>
              </div>
            </div>

            {/* Live Predicted Output Matrix */}
            <div style={{
              marginTop: 16,
              background: 'var(--bg-surface-sunken)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-xs)',
              padding: 16,
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 14,
              alignItems: 'center',
            }}>
              <div>
                <span className="stat-label">Predicted QPS</span>
                <div className="stat-number tabular-nums" style={{ fontSize: '1.4rem', color: 'var(--accent-glow)' }}>
                  {currentSweepPoint.qps.toLocaleString()}
                </div>
                <span className="badge-pill emerald" style={{ marginTop: 2 }}>
                  +{currentSweepPoint.qpsGainPct}% vs Base
                </span>
              </div>

              <div>
                <span className="stat-label">Predicted Recall@10</span>
                <div className="stat-number tabular-nums" style={{ fontSize: '1.4rem', color: 'var(--text-primary)' }}>
                  {(currentSweepPoint.recall * 100).toFixed(2)}%
                </div>
                <span className="badge-pill" style={{ marginTop: 2 }}>
                  -{currentSweepPoint.recallDropPct}% drop
                </span>
              </div>

              <div>
                <span className="stat-label">In-Degree Variance</span>
                <div className="stat-number tabular-nums" style={{ fontSize: '1.4rem', color: 'var(--semantic-emerald)' }}>
                  {currentMuPoint.inDegreeVariance}
                </div>
                <span className="badge-pill accent" style={{ marginTop: 2 }}>
                  -{currentMuPoint.varianceDropPct}% Var
                </span>
              </div>

              <div>
                <span className="stat-label">Evals / Query</span>
                <div className="stat-number tabular-nums" style={{ fontSize: '1.4rem', color: 'var(--text-secondary)' }}>
                  {currentSweepPoint.evals}
                </div>
                <span className="badge-pill" style={{ marginTop: 2 }}>
                  -30.1% compute
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Enlarged Figure Modal */}
      {zoomImage && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(8px)',
          zIndex: 200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 24,
        }} onClick={() => setZoomImage(null)}>
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-emphasis)',
            borderRadius: 'var(--radius-sm)',
            padding: 20,
            maxWidth: 960,
            width: '100%',
            position: 'relative',
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
                {zoomImage.title}
              </h3>
              <button
                onClick={() => setZoomImage(null)}
                className="btn-instrument"
                style={{ padding: '3px 7px' }}
              >
                <X size={15} />
              </button>
            </div>

            <div style={{
              background: '#ffffff',
              borderRadius: 'var(--radius-xs)',
              padding: 12,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              maxHeight: '65vh',
              overflow: 'hidden',
            }}>
              <img
                src={zoomImage.src}
                alt={zoomImage.title}
                style={{ width: '100%', height: 'auto', maxHeight: '60vh', objectFit: 'contain' }}
              />
            </div>

            <p style={{ marginTop: 12, fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {zoomImage.caption}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
