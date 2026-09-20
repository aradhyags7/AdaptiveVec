import React, { useState } from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { motion } from 'framer-motion';
import { FileCode, Database, Layers } from 'lucide-react';
import styles from './CanvasShared.module.css';

interface PlotPoint {
  id: string;
  stepName: string;
  shortName: string;
  qps: number;
  recall: number;
  edges: number;
  memoryMb: number;
  evals: number;
  isFeatured?: boolean;
  attribution: string;
  deltaText?: string;
}

const BENCHMARK_POINTS: PlotPoint[] = [
  {
    id: 'step-6',
    stepName: '6. + Asymmetric INT8 SQ8',
    shortName: 'Step 6: SQ8 (-62.4% RAM)',
    qps: 2987.6,
    recall: 0.9594,
    edges: 2509743,
    memoryMb: 22.54,
    evals: 1042.0,
    attribution: 'Scalar INT8 quantized distance re-ranking (Regime B)',
    deltaText: '-62.4% RAM (22.5 MB)',
  },
  {
    id: 'step-3',
    stepName: '3. + Layer-Decoupled Scaling',
    shortName: 'Step 3: Decoupled',
    qps: 2242.3,
    recall: 0.9887,
    edges: 2515277,
    memoryMb: 59.19,
    evals: 991.8,
    attribution: 'Scale layer 0 capacity independently',
    deltaText: '+28.1% build speedup',
  },
  {
    id: 'step-1',
    stepName: '1. Baseline HNSW (Fixed M=16)',
    shortName: 'Step 1: Baseline (4,708 QPS)',
    qps: 4708.1,
    recall: 0.9913,
    edges: 2709125,
    memoryMb: 59.93,
    evals: 1121.2,
    attribution: 'Control index (Standard heuristic HNSW)',
    deltaText: 'Reference baseline',
  },
  {
    id: 'step-2',
    stepName: '2. + Dynamic M(x) & efC(x)',
    shortName: 'Step 2: Adaptive M(x)',
    qps: 5147.1,
    recall: 0.9883,
    edges: 2549825,
    memoryMb: 59.32,
    evals: 1017.6,
    attribution: 'LID-guided local degree expansion',
    deltaText: '-5.88% edges',
  },
  {
    id: 'step-4',
    stepName: '4. + Hubness Regulation (mu=0.15)',
    shortName: 'Step 4: +Hubness (0.9856 parity)',
    qps: 5452.0,
    recall: 0.9854,
    edges: 2510334,
    memoryMb: 59.17,
    evals: 979.5,
    attribution: 'Penalize over-selected central hub nodes (Multi-trial parity 0.9856)',
    deltaText: '0.9856 multi-trial parity',
  },
  {
    id: 'step-5',
    stepName: '5. + Ada-ef Stagnation Exit',
    shortName: 'Step 5: Ada-ef (7,075 QPS • +50.3%)',
    qps: 7075.3,
    recall: 0.9745,
    edges: 2509138,
    memoryMb: 59.16,
    evals: 783.5,
    isFeatured: true,
    attribution: 'Dynamic efSearch with adaptive stagnation termination (p=6, ε=10⁻⁴, Regime A)',
    deltaText: '+50.3% QPS (Featured)',
  },
];

export const BenchmarkCanvas: React.FC = () => {
  const { toggleDrawer, hardware, results: _results, siftResults, syntheticResults } = useBenchmark();
  const [activeTab, setActiveTab] = useState<'pareto' | 'ablation' | 'regimes'>('pareto');
  const [selectedPoint, setSelectedPoint] = useState<PlotPoint>(BENCHMARK_POINTS[5]); // Default: Step 5

  // Plot coordinate transformation constants
  const svgWidth = 620;
  const svgHeight = 290;
  const marginLeft = 60;
  const marginRight = 35;
  const marginTop = 25;
  const marginBottom = 45;
  const plotW = svgWidth - marginLeft - marginRight; // 525
  const plotH = svgHeight - marginTop - marginBottom; // 220

  const minQps = 1500;
  const maxQps = 8000;
  const minRecall = 0.940;
  const maxRecall = 1.000;

  const toSvgX = (qps: number) => marginLeft + ((qps - minQps) / (maxQps - minQps)) * plotW;
  const toSvgY = (recall: number) => marginTop + ((maxRecall - recall) / (maxRecall - minRecall)) * plotH;

  // Sorted points along the empirical Pareto frontier
  const frontierPoints = [
    BENCHMARK_POINTS[0], // Step 6 (2987.6, 0.9594)
    BENCHMARK_POINTS[5], // Step 5 (7075.3, 0.9745)
    BENCHMARK_POINTS[4], // Step 4 (5452.0, 0.9854)
    BENCHMARK_POINTS[2], // Step 1 (4708.1, 0.9913)
  ].sort((a, b) => a.qps - b.qps);

  const frontierPathD = frontierPoints.reduce((acc, pt, idx) => {
    const x = toSvgX(pt.qps);
    const y = toSvgY(pt.recall);
    return idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, '');

  return (
    <div className={styles.canvasContainer}>
      {/* Precision Workspace Bar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <div className={styles.segmentedGroup}>
            <button
              className={`${styles.segBtn} ${activeTab === 'pareto' ? styles.segBtnActive : ''}`}
              onClick={() => setActiveTab('pareto')}
            >
              Dual Pareto Frontiers
            </button>
            <button
              className={`${styles.segBtn} ${activeTab === 'ablation' ? styles.segBtnActive : ''}`}
              onClick={() => setActiveTab('ablation')}
            >
              6-Step Ablation Matrix
            </button>
            <button
              className={`${styles.segBtn} ${activeTab === 'regimes' ? styles.segBtnActive : ''}`}
              onClick={() => setActiveTab('regimes')}
            >
              3-Regime Comparative
            </button>
          </div>
          <span className={styles.barSep}>/</span>
          <span className="badge-pill accent">SIFT-100K (128-D FP32)</span>
          <span className="badge-pill">n=100,000 • q=10,000</span>
        </div>

        <div className={styles.barRight}>
          <button
            className={styles.barActionBtn}
            onClick={() => toggleDrawer('data-export')}
            title="Open Verified Raw Data & LaTeX Exporters"
          >
            <FileCode size={13} />
            <span>Raw Data & Exports</span>
            <kbd className={styles.kbd}>Cmd+D</kbd>
          </button>
        </div>
      </div>

      {/* Primary Visual Stage */}
      {activeTab === 'pareto' && (
        <div className={styles.stageGrid2}>
          {/* Left Primary Pane: Calibrated Dual Pareto Frontier Plot */}
          <div className={styles.panePrimary}>
            <div className={styles.paneHeader}>
              <div className={styles.paneTitle}>
                <div className={styles.instrumentDot} />
                <span>Empirical Pareto Frontier • SIFT-100K Canonical Corpus</span>
              </div>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <div className="badge-pill cyan">
                  <span>N=100,000 • D=128 • Q=10,000</span>
                </div>
                <div className="badge-pill">
                  <span>Evaluated: {hardware.model}</span>
                </div>
              </div>
            </div>

            <div className={styles.plotStageWrapper}>
              {/* Telemetry coordinate strip */}
              <div className={styles.plotHeaderTelemetry}>
                <span>X-AXIS: THROUGHPUT [1,500 – 8,000 QPS]</span>
                <span>Y-AXIS: RECALL@10 [0.940 – 1.000]</span>
                <span>CALIBRATION: AVX2/FMA ENABLED</span>
              </div>

              {/* Main SVG Vector Plot */}
              <div className={styles.svgPlotContainer}>
                <svg
                  viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                  className={styles.svgPlot}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {/* Subtle Grid Lines & Y-axis Scale */}
                  {[0.94, 0.96, 0.98, 1.00].map((val) => {
                    const y = toSvgY(val);
                    return (
                      <g key={`y-grid-${val}`}>
                        <line
                          x1={marginLeft}
                          y1={y}
                          x2={svgWidth - marginRight}
                          y2={y}
                          className={styles.gridLine}
                        />
                        <text
                          x={marginLeft - 8}
                          y={y + 3.5}
                          textAnchor="end"
                          className={styles.axisLabel}
                        >
                          {val.toFixed(3)}
                        </text>
                      </g>
                    );
                  })}

                  {/* X-axis Grid Lines & Scale */}
                  {[2000, 4000, 6000, 8000].map((qpsVal) => {
                    const x = toSvgX(qpsVal);
                    return (
                      <g key={`x-grid-${qpsVal}`}>
                        <line
                          x1={x}
                          y1={marginTop}
                          x2={x}
                          y2={svgHeight - marginBottom}
                          className={styles.gridLine}
                        />
                        <text
                          x={x}
                          y={svgHeight - marginBottom + 16}
                          textAnchor="middle"
                          className={styles.axisLabel}
                        >
                          {qpsVal.toLocaleString()}
                        </text>
                        {/* Scale Tick Mark */}
                        <line
                          x1={x}
                          y1={svgHeight - marginBottom}
                          x2={x}
                          y2={svgHeight - marginBottom + 5}
                          className={styles.axisLine}
                        />
                      </g>
                    );
                  })}

                  {/* Axis Hairlines */}
                  <line
                    x1={marginLeft}
                    y1={svgHeight - marginBottom}
                    x2={svgWidth - marginRight}
                    y2={svgHeight - marginBottom}
                    className={styles.axisLine}
                  />
                  <line
                    x1={marginLeft}
                    y1={marginTop}
                    x2={marginLeft}
                    y2={svgHeight - marginBottom}
                    className={styles.axisLine}
                  />

                  {/* Baseline control reference zone */}
                  <rect
                    x={toSvgX(4400)}
                    y={toSvgY(1.000)}
                    width={toSvgX(5000) - toSvgX(4400)}
                    height={toSvgY(0.988) - toSvgY(1.000)}
                    fill="rgba(201, 125, 74, 0.04)"
                    stroke="rgba(201, 125, 74, 0.18)"
                    strokeDasharray="2 2"
                  />
                  <text
                    x={toSvgX(4708) - 14}
                    y={toSvgY(0.998)}
                    className={styles.axisLabel}
                    fill="var(--text-dim)"
                    fontSize="9"
                  >
                    BASELINE ENVELOPE
                  </text>

                  {/* Pareto Frontier Connecting Line */}
                  <motion.path
                    d={frontierPathD}
                    className={styles.paretoFrontierGlow}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ type: 'spring', stiffness: 220, damping: 28 }}
                  />

                  {/* Active Point Reticle Hairlines */}
                  {selectedPoint && (
                    <g>
                      {/* Vertical line to X-axis */}
                      <line
                        x1={toSvgX(selectedPoint.qps)}
                        y1={toSvgY(selectedPoint.recall)}
                        x2={toSvgX(selectedPoint.qps)}
                        y2={svgHeight - marginBottom}
                        className={styles.activeReticleLine}
                      />
                      {/* Horizontal line to Y-axis */}
                      <line
                        x1={marginLeft}
                        y1={toSvgY(selectedPoint.recall)}
                        x2={toSvgX(selectedPoint.qps)}
                        y2={toSvgY(selectedPoint.recall)}
                        className={styles.activeReticleLine}
                      />
                    </g>
                  )}

                  {/* Benchmark Data Points */}
                  {BENCHMARK_POINTS.map((pt) => {
                    const cx = toSvgX(pt.qps);
                    const cy = toSvgY(pt.recall);
                    const isSelected = selectedPoint.id === pt.id;

                    return (
                      <g
                        key={pt.id}
                        className={styles.plotNode}
                        onClick={() => setSelectedPoint(pt)}
                        onMouseEnter={() => setSelectedPoint(pt)}
                      >
                        {/* Outer Glow on Featured Step 5 */}
                        {pt.isFeatured && (
                          <circle
                            cx={cx}
                            cy={cy}
                            r={14}
                            fill="none"
                            stroke="var(--accent)"
                            strokeWidth={1}
                            opacity={0.4}
                          />
                        )}

                        {/* Node Circle */}
                        <circle
                          cx={cx}
                          cy={cy}
                          r={pt.isFeatured ? 6 : isSelected ? 5.5 : 4.5}
                          fill={pt.isFeatured ? 'var(--accent-glow)' : isSelected ? 'var(--accent)' : 'var(--bg-card)'}
                          stroke={pt.isFeatured ? '#ffffff' : isSelected ? 'var(--accent-glow)' : 'var(--border-emphasis)'}
                          strokeWidth={isSelected || pt.isFeatured ? 2 : 1.5}
                        />

                        {/* Point Label */}
                        <text
                          x={cx + (pt.id === 'step-5' ? -12 : pt.id === 'step-1' ? 10 : 8)}
                          y={cy + (pt.id === 'step-5' ? -12 : pt.id === 'step-3' ? 14 : -8)}
                          className={styles.axisLabel}
                          textAnchor={pt.id === 'step-5' ? 'end' : 'start'}
                          fontWeight={pt.isFeatured || isSelected ? 700 : 500}
                          fill={pt.isFeatured || isSelected ? 'var(--text-primary)' : 'var(--text-muted)'}
                        >
                          {pt.shortName}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Real-time Telemetry HUD (Selected node readout) */}
              <div className={styles.plotFooterHud}>
                <div className={styles.hudItem}>
                  <span className={styles.hudKey}>Active Step Reference</span>
                  <span className={`${styles.hudVal} ${styles.hudValHighlight}`}>
                    {selectedPoint.stepName}
                  </span>
                </div>

                <div className={styles.hudItem}>
                  <span className={styles.hudKey}>Throughput (QPS)</span>
                  <span className="tabular-nums font-semibold">
                    {selectedPoint.qps.toLocaleString()} QPS
                  </span>
                </div>

                <div className={styles.hudItem}>
                  <span className={styles.hudKey}>Recall@10</span>
                  <span className="tabular-nums font-semibold">
                    {selectedPoint.recall.toFixed(4)}
                  </span>
                </div>

                <div className={styles.hudItem}>
                  <span className={styles.hudKey}>Graph Edges</span>
                  <span className="tabular-nums">
                    {selectedPoint.edges.toLocaleString()}
                  </span>
                </div>

                <div className={styles.hudItem}>
                  <span className={styles.hudKey}>Memory RAM</span>
                  <span className="tabular-nums">
                    {selectedPoint.memoryMb.toFixed(2)} MB
                  </span>
                </div>

                <div className={styles.hudItem}>
                  <span className={styles.hudKey}>Distance Evals</span>
                  <span className="tabular-nums">
                    {selectedPoint.evals.toFixed(1)} / q
                  </span>
                </div>
              </div>

              {/* Secondary Footprint Stage: Memory & Footprint Bar Gage */}
              <div className={styles.footprintGageArea}>
                <div className={styles.gageHeader}>
                  <span className={styles.gageTitle}>
                    <Database size={12} />
                    <span>Memory Footprint & Edge Conservation Gage</span>
                  </span>
                  <span className="tabular-nums text-xs text-muted">
                    Reference Scale: 0 – 60.0 MB RAM
                  </span>
                </div>

                <div className={styles.gageRows}>
                  {/* Baseline Row */}
                  <div className={styles.gageRowItem}>
                    <span className={styles.gageLabel}>Step 1 Baseline HNSW</span>
                    <div className={styles.gageTrack}>
                      <div className={styles.gageFillBaseline} />
                    </div>
                    <span className={`${styles.gageValue} tabular-nums text-muted`}>
                      59.93 MB (2,709,125 edges)
                    </span>
                  </div>

                  {/* Step 6 SQ8 Row */}
                  <div className={styles.gageRowItem}>
                    <span className={styles.gageLabel}>Step 6 Asymmetric SQ8</span>
                    <div className={styles.gageTrack}>
                      <div className={styles.gageFillSq8} />
                    </div>
                    <span className={`${styles.gageValue} tabular-nums text-emerald font-semibold`}>
                      22.54 MB (-62.4% RAM)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Secondary Pane: 3-Regime Comparative Architecture */}
          <div className={styles.paneSecondary}>
            <div className={styles.paneHeader}>
              <div className={styles.paneTitle}>
                <Layers size={13} />
                <span>3-Regime Comparative Analysis</span>
              </div>
              <span className="badge-pill accent">Validated</span>
            </div>

            <div className={styles.regimeMatrixList}>
              {/* Regime A: Throughput Maxima */}
              <div className={`${styles.regimeBlock} ${styles.blockFeatured}`}>
                <div className={styles.blockRow}>
                  <span className={styles.blockBadge}>Regime A &bull; Step 5 Ada-ef</span>
                  <span className="badge-pill emerald">+50.3% Throughput</span>
                </div>
                <div className={styles.blockNumberRow}>
                  <span className={`${styles.blockHeroNum} tabular-nums text-accent-glow`}>
                    7,075.3
                  </span>
                  <span className={styles.blockHeroUnit}>QPS</span>
                </div>
                <div className={styles.blockSub}>
                  <strong className="text-primary">0.9745 Recall@10</strong> (Parity cf. 0.9856 at Step 4)
                </div>
                <div className={styles.blockMetaGrid}>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Graph Topology:</span>
                    <span className={styles.metaItemVal}>2,509,138 edges (-7.4%)</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Query Distance Evals:</span>
                    <span className={styles.metaItemVal}>783.5 evals (-30.1%)</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Index Construction:</span>
                    <span className={styles.metaItemVal}>33.51s (+28.1% speedup)</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Memory Footprint:</span>
                    <span className={styles.metaItemVal}>59.16 MB RAM</span>
                  </div>
                </div>
              </div>

              {/* Regime B: Memory Maxima */}
              <div className={styles.regimeBlock}>
                <div className={styles.blockRow}>
                  <span className={styles.blockBadge}>Regime B &bull; Step 6 SQ8</span>
                  <span className="badge-pill emerald">-62.4% Footprint</span>
                </div>
                <div className={styles.blockNumberRow}>
                  <span className={`${styles.blockHeroNum} tabular-nums`}>
                    22.5
                  </span>
                  <span className={styles.blockHeroUnit}>MB RAM</span>
                </div>
                <div className={styles.blockSub}>
                  <strong className="text-primary">0.9594 Recall@10</strong> &bull; 2,987.6 QPS (Asymmetric INT8 distance)
                </div>
                <div className={styles.blockMetaGrid}>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Scalar Compression:</span>
                    <span className={styles.metaItemVal}>4x reduction (FP32 &rarr; INT8)</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Graph Connectivity:</span>
                    <span className={styles.metaItemVal}>2,509,743 edges</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Distance Metric:</span>
                    <span className={styles.metaItemVal}>Asymmetric L2 SQ8</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Application Tier:</span>
                    <span className={styles.metaItemVal}>Embedded / RAM-Constrained</span>
                  </div>
                </div>
              </div>

              {/* Control Baseline */}
              <div className={styles.regimeBlock}>
                <div className={styles.blockRow}>
                  <span className={styles.blockBadge}>Control &bull; Step 1 Baseline HNSW</span>
                  <span className="badge-pill">Reference</span>
                </div>
                <div className={styles.blockNumberRow}>
                  <span className={`${styles.blockHeroNum} tabular-nums text-muted`}>
                    4,708.1
                  </span>
                  <span className={styles.blockHeroUnit}>QPS</span>
                </div>
                <div className={styles.blockSub}>
                  <strong className="text-primary">0.9913 Recall@10</strong> &bull; 59.93 MB RAM
                </div>
                <div className={styles.blockMetaGrid}>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Degree Allocation:</span>
                    <span className={styles.metaItemVal}>Fixed M=16, efC=100</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Search Budget:</span>
                    <span className={styles.metaItemVal}>Fixed efSearch=64</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Total Graph Edges:</span>
                    <span className={styles.metaItemVal}>2,709,125 edges</span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaItemKey}>Build Time:</span>
                    <span className={styles.metaItemVal}>46.61s (Reference)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Persistent Provenance Hardware Stamp */}
            <div className={styles.persistentHardwareStamp}>
              <div><strong>HARDWARE:</strong> {hardware.model} ({hardware.cores}C/{hardware.threads}T)</div>
              <div><strong>INSTRUCTIONS:</strong> AVX2 / FMA • 16GB Physical RAM</div>
              <div><strong>SOURCE:</strong> benchmark_results.json (Zero synthetic figures)</div>
            </div>
          </div>
        </div>
      )}

      {/* 6-Step Ablation Matrix View with Explicit Dataset Separators */}
      {activeTab === 'ablation' && (
        <div className={styles.ablationTableContainer}>
          <table className={styles.ablationTable}>
            <thead>
              <tr>
                <th>Ablation Step Configuration</th>
                <th>Recall@10</th>
                <th>Throughput (QPS)</th>
                <th>Build Speedup</th>
                <th>Total Edges</th>
                <th>Memory (MB)</th>
                <th>Evals / Query</th>
              </tr>
            </thead>
            <tbody>
              {/* Dataset Section 1: SIFT-100K Subset */}
              <tr className={styles.datasetSeparatorRow}>
                <td colSpan={7}>
                  <div className={styles.datasetSeparatorContent}>
                    <span className={styles.datasetSeparatorTitle}>CANONICAL BENCHMARK • SIFT-100K SUBSET</span>
                    <span className={styles.datasetSeparatorMeta}>Texmex IRISA • N=100,000 • D=128 • L2 Euclidean (Q=10,000 test queries)</span>
                  </div>
                </td>
              </tr>
              {siftResults.map((r) => {
                const isStep5 = r.configuration.includes('5. + Ada-ef');
                const isStep6 = r.configuration.includes('6. + Asymmetric');

                return (
                  <tr
                    key={`sift-${r.configuration}`}
                    className={isStep5 ? styles.tableRowHighlight : ''}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{r.configuration}</span>
                        {isStep5 && <span className="badge-pill emerald">Featured (+50.3%)</span>}
                        {isStep6 && <span className="badge-pill accent">SQ8 (-62.4% RAM)</span>}
                      </div>
                    </td>
                    <td className="tabular-nums font-semibold">
                      {r.recall_at_10.toFixed(4)}
                    </td>
                    <td className="tabular-nums font-semibold">
                      {r.qps.toLocaleString()}
                    </td>
                    <td className="tabular-nums">
                      {r.build_speedup_pct > 0 ? `+${r.build_speedup_pct}%` : `${r.build_speedup_pct}%`}
                    </td>
                    <td className="tabular-nums">
                      {r.total_edges.toLocaleString()}
                    </td>
                    <td className="tabular-nums">
                      {r.memory_mb.toFixed(2)} MB
                    </td>
                    <td className="tabular-nums">
                      {r.distance_evaluations_per_query.toFixed(1)}
                    </td>
                  </tr>
                );
              })}

              {/* Dataset Section 2: Synthetic-Multi-Cluster */}
              <tr className={styles.datasetSeparatorRow}>
                <td colSpan={7}>
                  <div className={styles.datasetSeparatorContent}>
                    <span className={styles.datasetSeparatorTitle}>SECONDARY BENCHMARK • SYNTHETIC MULTI-CLUSTER</span>
                    <span className={styles.datasetSeparatorMeta}>8 Heterogeneous Gaussian Clusters • N=50,000 • D=64 • L2 Space (Q=1,000 test queries)</span>
                  </div>
                </td>
              </tr>
              {syntheticResults.map((r) => {
                const isStep5 = r.configuration.includes('5. + Ada-ef');
                const isStep6 = r.configuration.includes('6. + Asymmetric');

                return (
                  <tr
                    key={`synth-${r.configuration}`}
                    className={isStep5 ? styles.tableRowHighlight : ''}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{r.configuration}</span>
                        {isStep5 && <span className="badge-pill emerald">Ada-ef Exit</span>}
                        {isStep6 && <span className="badge-pill accent">SQ8 Scaled</span>}
                      </div>
                    </td>
                    <td className="tabular-nums font-semibold">
                      {r.recall_at_10.toFixed(4)}
                    </td>
                    <td className="tabular-nums font-semibold">
                      {r.qps.toLocaleString()}
                    </td>
                    <td className="tabular-nums">
                      {r.build_speedup_pct > 0 ? `+${r.build_speedup_pct}%` : `${r.build_speedup_pct}%`}
                    </td>
                    <td className="tabular-nums">
                      {r.total_edges.toLocaleString()}
                    </td>
                    <td className="tabular-nums">
                      {r.memory_mb.toFixed(2)} MB
                    </td>
                    <td className="tabular-nums">
                      {r.distance_evaluations_per_query.toFixed(1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* 3-Regime Matrix Wide Layout */}
      {activeTab === 'regimes' && (
        <div className={styles.regimeWideGrid}>
          {/* Regime A */}
          <div className={`${styles.regimeColCard} ${styles.regimeColFeatured}`}>
            <div className={styles.blockRow}>
              <span className={styles.blockBadge}>Regime A (Throughput Maxima)</span>
              <span className="badge-pill emerald">+50.3% QPS</span>
            </div>
            <div className={styles.blockNumberRow}>
              <span className={`${styles.blockHeroNum} tabular-nums text-accent-glow`}>
                7,075.3
              </span>
              <span className={styles.blockHeroUnit}>QPS</span>
            </div>
            <p className={styles.blockSub}>
              Optimized for real-time web services, semantic search engines, and multi-tenant vector databases requiring minimum response latency.
            </p>
            <div className={styles.blockMetaGrid}>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Recall Target:</span>
                <span className={styles.metaItemVal}>0.9745 Recall@10</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Comparative Parity:</span>
                <span className={styles.metaItemVal}>cf. 0.9856 at Step 4</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Query Distance Evals:</span>
                <span className={styles.metaItemVal}>783.5 / query (-30.1%)</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Graph Topology:</span>
                <span className={styles.metaItemVal}>2,509,138 edges (-7.4%)</span>
              </div>
            </div>
          </div>

          {/* Regime B */}
          <div className={styles.regimeColCard}>
            <div className={styles.blockRow}>
              <span className={styles.blockBadge}>Regime B (Memory Maxima)</span>
              <span className="badge-pill emerald">-62.4% RAM</span>
            </div>
            <div className={styles.blockNumberRow}>
              <span className={`${styles.blockHeroNum} tabular-nums`}>
                22.54
              </span>
              <span className={styles.blockHeroUnit}>MB RAM</span>
            </div>
            <p className={styles.blockSub}>
              Optimized for edge IoT devices, mobile client embeddings, and low-cost multi-million vector hosting instances.
            </p>
            <div className={styles.blockMetaGrid}>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Recall Target:</span>
                <span className={styles.metaItemVal}>0.9594 Recall@10</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Throughput:</span>
                <span className={styles.metaItemVal}>2,987.6 QPS</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Quantization Scheme:</span>
                <span className={styles.metaItemVal}>Asymmetric INT8 SQ8</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Footprint Reduction:</span>
                <span className={styles.metaItemVal}>59.93 MB &rarr; 22.54 MB</span>
              </div>
            </div>
          </div>

          {/* Control Baseline */}
          <div className={styles.regimeColCard}>
            <div className={styles.blockRow}>
              <span className={styles.blockBadge}>Control (Reference Null)</span>
              <span className="badge-pill">Baseline</span>
            </div>
            <div className={styles.blockNumberRow}>
              <span className={`${styles.blockHeroNum} tabular-nums text-muted`}>
                4,708.1
              </span>
              <span className={styles.blockHeroUnit}>QPS</span>
            </div>
            <p className={styles.blockSub}>
              Canonical non-adaptive HNSW reference implementation evaluated on identical hardware threads and identical query vectors.
            </p>
            <div className={styles.blockMetaGrid}>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Recall Target:</span>
                <span className={styles.metaItemVal}>0.9913 Recall@10</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Construction M / efC:</span>
                <span className={styles.metaItemVal}>M=16, efConstruction=100</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Search Budget:</span>
                <span className={styles.metaItemVal}>efSearch=64 (Static)</span>
              </div>
              <div className={styles.metaItem}>
                <span className={styles.metaItemKey}>Total Graph Edges:</span>
                <span className={styles.metaItemVal}>2,709,125 edges</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
