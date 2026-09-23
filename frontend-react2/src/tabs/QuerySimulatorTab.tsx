import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, SkipForward, CheckCircle2, Activity } from 'lucide-react';

interface SimulationHop {
  hop: number;
  nodeId: number;
  x: number;
  y: number;
  dist: number;
  deltaDist: number;
  isStagnant: boolean;
  stagnationCounter: number;
}

const SIMULATED_HOPS: SimulationHop[] = [
  { hop: 0, nodeId: 10402, x: 220, y: 110, dist: 0.842, deltaDist: 0.000, isStagnant: false, stagnationCounter: 0 },
  { hop: 1, nodeId: 12890, x: 260, y: 140, dist: 0.612, deltaDist: 0.230, isStagnant: false, stagnationCounter: 0 },
  { hop: 2, nodeId: 18450, x: 310, y: 180, dist: 0.450, deltaDist: 0.162, isStagnant: false, stagnationCounter: 0 },
  { hop: 3, nodeId: 24102, x: 380, y: 220, dist: 0.315, deltaDist: 0.135, isStagnant: false, stagnationCounter: 0 },
  { hop: 4, nodeId: 31050, x: 440, y: 250, dist: 0.220, deltaDist: 0.095, isStagnant: false, stagnationCounter: 0 },
  { hop: 5, nodeId: 36200, x: 490, y: 270, dist: 0.165, deltaDist: 0.055, isStagnant: false, stagnationCounter: 0 },
  { hop: 6, nodeId: 40150, x: 530, y: 285, dist: 0.132, deltaDist: 0.033, isStagnant: false, stagnationCounter: 0 },
  { hop: 7, nodeId: 43200, x: 560, y: 295, dist: 0.110, deltaDist: 0.022, isStagnant: false, stagnationCounter: 0 },
  { hop: 8, nodeId: 45100, x: 580, y: 302, dist: 0.098, deltaDist: 0.012, isStagnant: false, stagnationCounter: 0 },
  { hop: 9, nodeId: 46500, x: 595, y: 308, dist: 0.091, deltaDist: 0.007, isStagnant: false, stagnationCounter: 0 },
  { hop: 10, nodeId: 47200, x: 605, y: 312, dist: 0.088, deltaDist: 0.003, isStagnant: false, stagnationCounter: 0 },
  // Stagnation threshold delta < 0.0001 begins
  { hop: 11, nodeId: 47800, x: 612, y: 314, dist: 0.08795, deltaDist: 0.00005, isStagnant: true, stagnationCounter: 1 },
  { hop: 12, nodeId: 48010, x: 615, y: 315, dist: 0.08792, deltaDist: 0.00003, isStagnant: true, stagnationCounter: 2 },
  { hop: 13, nodeId: 48120, x: 617, y: 316, dist: 0.08791, deltaDist: 0.00001, isStagnant: true, stagnationCounter: 3 },
  { hop: 14, nodeId: 48190, x: 618, y: 316, dist: 0.08791, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 4 },
  { hop: 15, nodeId: 48210, x: 619, y: 317, dist: 0.08790, deltaDist: 0.00001, isStagnant: true, stagnationCounter: 5 },
  // Hop 16: p=6 threshold reached! Early exit fires!
  { hop: 16, nodeId: 48219, x: 620, y: 317, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 6 },
  // Standard HNSW continues needlessly:
  { hop: 17, nodeId: 48310, x: 621, y: 318, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 7 },
  { hop: 18, nodeId: 48420, x: 622, y: 317, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 8 },
  { hop: 19, nodeId: 48500, x: 620, y: 319, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 9 },
  { hop: 20, nodeId: 48610, x: 623, y: 316, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 10 },
  { hop: 21, nodeId: 48700, x: 621, y: 318, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 11 },
  { hop: 22, nodeId: 48800, x: 619, y: 317, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 12 },
  { hop: 23, nodeId: 48900, x: 620, y: 318, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 13 },
  { hop: 24, nodeId: 49000, x: 620, y: 317, dist: 0.08790, deltaDist: 0.00000, isStagnant: true, stagnationCounter: 14 },
];

export const QuerySimulatorTab: React.FC = () => {
  const [currentHopIndex, setCurrentHopIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [showAdaptiveExit, setShowAdaptiveExit] = useState<boolean>(true);

  const currentHop = SIMULATED_HOPS[currentHopIndex];
  const isAdaptiveHalted = showAdaptiveExit && currentHopIndex >= 16;

  // Auto-play timer
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentHopIndex((prev) => {
          if (showAdaptiveExit && prev >= 16) {
            setIsPlaying(false);
            return 16;
          }
          if (prev >= SIMULATED_HOPS.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 350);
    }
    return () => clearInterval(timer);
  }, [isPlaying, showAdaptiveExit]);

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentHopIndex(0);
  };

  const handleStep = () => {
    setIsPlaying(false);
    setCurrentHopIndex((prev) => Math.min(prev + 1, SIMULATED_HOPS.length - 1));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className="badge-pill accent">Dynamic Trajectory Telemetry</span>
            <span className="badge-pill">Greedy Beam Search</span>
          </div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>
            Query Trajectory & Distance Stagnation Observatory
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            Observing distance delta saturation (p=6, ε=10⁻⁴) to terminate greedy routing 8 hops ahead of standard HNSW.
          </p>
        </div>

        {/* Playback Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            className={`btn-instrument ${isPlaying ? '' : 'primary'}`}
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            <span>{isPlaying ? 'Pause' : 'Play Trajectory'}</span>
          </button>

          <button className="btn-instrument" onClick={handleStep} title="Step forward 1 hop">
            <SkipForward size={13} />
            <span>Step</span>
          </button>

          <button className="btn-instrument" onClick={handleReset} title="Reset to hop 0">
            <RotateCcw size={13} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Main Simulation Stage: 2-Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
        {/* Left Column: 2D Spatial Vector Trajectory Canvas */}
        <div className="instrument-card" style={{ padding: 16, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Vector Manifold Routing Projection
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <span className="badge-pill">Layer 3 → Layer 0</span>
              <button
                className={`badge-pill ${showAdaptiveExit ? 'emerald' : 'amber'}`}
                style={{ cursor: 'pointer' }}
                onClick={() => setShowAdaptiveExit(!showAdaptiveExit)}
              >
                {showAdaptiveExit ? '✓ Adaptive Exit Active' : 'Standard HNSW (Exhaustive)'}
              </button>
            </div>
          </div>

          {/* SVG Map Canvas */}
          <div style={{
            background: 'var(--canvas-bg)',
            borderRadius: 'var(--radius-xs)',
            border: '1px solid var(--border-default)',
            position: 'relative',
            height: 360,
            overflow: 'hidden',
          }}>
            <svg viewBox="0 0 800 400" style={{ width: '100%', height: '100%' }}>
              <defs>
                <radialGradient id="denseClusterAmber" cx="75%" cy="80%" r="40%">
                  <stop offset="0%" stopColor="rgba(201, 125, 74, 0.2)" />
                  <stop offset="100%" stopColor="transparent" />
                </radialGradient>
                <radialGradient id="sparseRidgeSubtle" cx="25%" cy="30%" r="35%">
                  <stop offset="0%" stopColor="rgba(225, 210, 190, 0.08)" />
                  <stop offset="100%" stopColor="transparent" />
                </radialGradient>
              </defs>

              {/* Manifold Density Contours */}
              <circle cx="620" cy="320" r="140" fill="url(#denseClusterAmber)" />
              <circle cx="240" cy="120" r="110" fill="url(#sparseRidgeSubtle)" />

              <text x="210" y="70" fill="var(--text-dim)" fontSize="10.5" fontWeight="600" fontFamily="var(--font-mono)">
                Sparse Ridge (High LID ~24)
              </text>
              <text x="520" y="380" fill="var(--text-dim)" fontSize="10.5" fontWeight="600" fontFamily="var(--font-mono)">
                Dense Core Cluster (Low LID ~8, Hub Regulated)
              </text>

              {/* Background ambient vectors */}
              {[
                [180, 100], [210, 130], [250, 90], [290, 150], [340, 170],
                [420, 230], [470, 260], [510, 240], [540, 310], [580, 270],
                [640, 330], [670, 310], [630, 360], [590, 340], [660, 350],
              ].map(([bx, by], i) => (
                <circle key={i} cx={bx} cy={by} r="2.5" fill="var(--border-emphasis)" />
              ))}

              {/* Trajectory Polyline */}
              {currentHopIndex > 0 && (
                <polyline
                  points={SIMULATED_HOPS.slice(0, currentHopIndex + 1).map((h) => `${h.x},${h.y}`).join(' ')}
                  fill="none"
                  stroke={isAdaptiveHalted ? 'var(--semantic-emerald)' : 'var(--accent-glow)'}
                  strokeWidth="2"
                />
              )}

              {/* Visited Hops */}
              {SIMULATED_HOPS.slice(0, currentHopIndex + 1).map((h, idx) => {
                const isExit = idx === 16;
                const isCurrent = idx === currentHopIndex;

                return (
                  <g key={idx}>
                    <circle
                      cx={h.x}
                      cy={h.y}
                      r={isCurrent ? 6 : isExit ? 5 : 3.5}
                      fill={isExit ? 'var(--semantic-emerald)' : isCurrent ? 'var(--accent-glow)' : 'var(--accent)'}
                      stroke="#ffffff"
                      strokeWidth={isCurrent ? 2 : 1}
                    />
                    {isCurrent && (
                      <text x={h.x + 8} y={h.y - 6} fill="var(--accent-glow)" fontSize="10.5" fontWeight="700" fontFamily="var(--font-mono)">
                        Hop #{h.hop} (Node {h.nodeId})
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Target Nearest Neighbor */}
              <circle cx="620" cy="317" r="8" fill="none" stroke="var(--semantic-emerald)" strokeWidth="1.5" strokeDasharray="3 3" />
              <text x="635" y="321" fill="var(--semantic-emerald)" fontSize="10.5" fontWeight="700" fontFamily="var(--font-sans)">
                True 1-NN Vector #48219
              </text>
            </svg>
          </div>
        </div>

        {/* Right Column: Real-Time Telemetry & Distance Stagnation Gauge */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Current Hop Telemetry */}
          <div className="instrument-card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <span className="stat-label">Routing Step Telemetry</span>
              <span className="badge-pill accent tabular-nums">Hop #{currentHop.hop} / 24</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div style={{ background: 'var(--bg-surface-sunken)', padding: 10, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>Current Best Distance</div>
                <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {currentHop.dist.toFixed(5)}
                </div>
              </div>

              <div style={{ background: 'var(--bg-surface-sunken)', padding: 10, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>Improvement Δd</div>
                <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 700, color: currentHop.deltaDist < 0.0001 ? 'var(--semantic-amber)' : 'var(--semantic-emerald)', marginTop: 2 }}>
                  {currentHop.deltaDist.toFixed(5)}
                </div>
              </div>
            </div>

            {/* Stagnation Window Progress Bar */}
            <div style={{ marginTop: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', marginBottom: 4 }}>
                <span style={{ color: 'var(--text-muted)' }}>Stagnation Counter (p = 6 threshold)</span>
                <span className="tabular-nums" style={{ color: currentHop.stagnationCounter >= 6 ? 'var(--semantic-emerald)' : 'var(--accent-glow)', fontWeight: 700 }}>
                  {Math.min(currentHop.stagnationCounter, 6)} / 6
                </span>
              </div>
              <div style={{ width: '100%', height: 6, background: 'var(--bg-app)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  width: `${(Math.min(currentHop.stagnationCounter, 6) / 6) * 100}%`,
                  height: '100%',
                  background: currentHop.stagnationCounter >= 6 ? 'var(--semantic-emerald)' : 'var(--accent)',
                  transition: 'width 0.15s ease',
                }} />
              </div>
            </div>
          </div>

          {/* Stagnation Trigger Banner */}
          {currentHop.stagnationCounter >= 6 ? (
            <div style={{
              background: 'var(--semantic-emerald-bg)',
              border: '1px solid rgba(107, 163, 126, 0.4)',
              borderRadius: 'var(--radius-xs)',
              padding: 14,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}>
              <CheckCircle2 size={18} color="var(--semantic-emerald)" style={{ marginTop: 1, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--semantic-emerald)' }}>
                  Distance Stagnation Early Exit Triggered!
                </div>
                <p style={{ fontSize: '11px', color: 'var(--text-primary)', lineHeight: 1.5, marginTop: 3 }}>
                  Consecutive distance improvement dropped below ε = 10⁻⁴ across 6 hops. True 1-NN <strong>#48219</strong> already identified. Terminating search now saves <strong>338 redundant evaluations (-30.1% compute)</strong> with 0% recall loss.
                </p>
              </div>
            </div>
          ) : (
            <div style={{
              background: 'var(--bg-card-subtle)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-xs)',
              padding: 14,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}>
              <Activity size={18} color="var(--accent-glow)" style={{ marginTop: 1, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Greedy Graph Traversal in Progress
                </div>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5, marginTop: 3 }}>
                  Navigating down from upper express layers to ground layer 0. Tracking Euclidean distance deltas relative to query point.
                </p>
              </div>
            </div>
          )}

          {/* Comparative Summary Table */}
          <div className="instrument-card" style={{ padding: 14, fontSize: '11.5px' }}>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
              Execution Strategy Comparison:
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Standard HNSW:</span>
              <span className="tabular-nums" style={{ color: 'var(--semantic-amber)' }}>24 Hops • 1,121.2 Evals</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-muted)' }}>AdaptiveVec (Regime A):</span>
              <span className="tabular-nums" style={{ color: 'var(--semantic-emerald)', fontWeight: 700 }}>16 Hops • 783.5 Evals (-30.1%)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0' }}>
              <span style={{ color: 'var(--text-muted)' }}>Throughput Delta:</span>
              <span className="tabular-nums" style={{ color: 'var(--accent-glow)', fontWeight: 700 }}>+50.3% Search QPS</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
