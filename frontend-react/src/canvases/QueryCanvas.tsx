import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { Play, Pause, SkipForward, RotateCcw, Activity } from 'lucide-react';
import styles from './CanvasShared.module.css';

interface HopData {
  hop: number;
  nodeId: number;
  layer: number;
  distance: number;
  deltaDist: number;
  isStagnant: boolean;
  candidates: { id: number; dist: number }[];
}

const QUERY_TRACES: Record<string, { name: string; description: string; exitHop: number; evals: number; hops: HopData[] }> = {
  'q0': {
    name: 'Query #0 (Dense Core)',
    description: 'High-density cluster vector exhibiting rapid centroid convergence and adaptive stagnation termination.',
    exitHop: 16,
    evals: 783.5,
    hops: [
      { hop: 0, nodeId: 10402, layer: 3, distance: 0.842, deltaDist: 0.0000, isStagnant: false, candidates: [{ id: 10402, dist: 0.842 }, { id: 11200, dist: 0.889 }, { id: 14001, dist: 0.912 }] },
      { hop: 1, nodeId: 12890, layer: 2, distance: 0.612, deltaDist: 0.2300, isStagnant: false, candidates: [{ id: 12890, dist: 0.612 }, { id: 13502, dist: 0.645 }, { id: 10402, dist: 0.842 }] },
      { hop: 2, nodeId: 18450, layer: 1, distance: 0.450, deltaDist: 0.1620, isStagnant: false, candidates: [{ id: 18450, dist: 0.450 }, { id: 19120, dist: 0.478 }, { id: 12890, dist: 0.612 }] },
      { hop: 3, nodeId: 24102, layer: 0, distance: 0.315, deltaDist: 0.1350, isStagnant: false, candidates: [{ id: 24102, dist: 0.315 }, { id: 25001, dist: 0.342 }, { id: 18450, dist: 0.450 }] },
      { hop: 4, nodeId: 31050, layer: 0, distance: 0.220, deltaDist: 0.0950, isStagnant: false, candidates: [{ id: 31050, dist: 0.220 }, { id: 31502, dist: 0.235 }, { id: 24102, dist: 0.315 }] },
      { hop: 5, nodeId: 36200, layer: 0, distance: 0.165, deltaDist: 0.0550, isStagnant: false, candidates: [{ id: 36200, dist: 0.165 }, { id: 36801, dist: 0.178 }, { id: 31050, dist: 0.220 }] },
      { hop: 6, nodeId: 40150, layer: 0, distance: 0.132, deltaDist: 0.0330, isStagnant: false, candidates: [{ id: 40150, dist: 0.132 }, { id: 40502, dist: 0.141 }, { id: 36200, dist: 0.165 }] },
      { hop: 7, nodeId: 43200, layer: 0, distance: 0.110, deltaDist: 0.0220, isStagnant: false, candidates: [{ id: 43200, dist: 0.110 }, { id: 43601, dist: 0.118 }, { id: 40150, dist: 0.132 }] },
      { hop: 8, nodeId: 45100, layer: 0, distance: 0.098, deltaDist: 0.0120, isStagnant: false, candidates: [{ id: 45100, dist: 0.098 }, { id: 45402, dist: 0.102 }, { id: 43200, dist: 0.110 }] },
      { hop: 9, nodeId: 46500, layer: 0, distance: 0.091, deltaDist: 0.0070, isStagnant: false, candidates: [{ id: 46500, dist: 0.091 }, { id: 46801, dist: 0.094 }, { id: 45100, dist: 0.098 }] },
      { hop: 10, nodeId: 47200, layer: 0, distance: 0.088, deltaDist: 0.0030, isStagnant: false, candidates: [{ id: 47200, dist: 0.088 }, { id: 47502, dist: 0.089 }, { id: 46500, dist: 0.091 }] },
      // Stagnation window begins at Hop 11 (p=1)
      { hop: 11, nodeId: 47800, layer: 0, distance: 0.08795, deltaDist: 0.00005, isStagnant: true, candidates: [{ id: 47800, dist: 0.08795 }, { id: 47200, dist: 0.0880 }, { id: 47901, dist: 0.0882 }] },
      { hop: 12, nodeId: 48010, layer: 0, distance: 0.08792, deltaDist: 0.00003, isStagnant: true, candidates: [{ id: 48010, dist: 0.08792 }, { id: 47800, dist: 0.08795 }, { id: 48102, dist: 0.0881 }] },
      { hop: 13, nodeId: 48120, layer: 0, distance: 0.08791, deltaDist: 0.00001, isStagnant: true, candidates: [{ id: 48120, dist: 0.08791 }, { id: 48010, dist: 0.08792 }, { id: 48150, dist: 0.0880 }] },
      { hop: 14, nodeId: 48190, layer: 0, distance: 0.08791, deltaDist: 0.00000, isStagnant: true, candidates: [{ id: 48190, dist: 0.08791 }, { id: 48120, dist: 0.08791 }, { id: 48201, dist: 0.0880 }] },
      { hop: 15, nodeId: 48210, layer: 0, distance: 0.08790, deltaDist: 0.00001, isStagnant: true, candidates: [{ id: 48210, dist: 0.08790 }, { id: 48190, dist: 0.08791 }, { id: 48215, dist: 0.0880 }] },
      // Hop 16: p=6 threshold reached (ε = 10⁻⁴). Stagnation early exit fired!
      { hop: 16, nodeId: 48219, layer: 0, distance: 0.08790, deltaDist: 0.00000, isStagnant: true, candidates: [{ id: 48219, dist: 0.08790 }, { id: 48210, dist: 0.08790 }, { id: 48225, dist: 0.08798 }] },
    ]
  },
  'q1': {
    name: 'Query #1 (Sparse Ridge)',
    description: 'Sparse high-LID ridge search requiring extended exploration depth to preserve recall.',
    exitHop: 21,
    evals: 1018.4,
    hops: [
      { hop: 0, nodeId: 5012, layer: 3, distance: 0.912, deltaDist: 0.000, isStagnant: false, candidates: [{ id: 5012, dist: 0.912 }] },
      { hop: 1, nodeId: 8120, layer: 2, distance: 0.742, deltaDist: 0.170, isStagnant: false, candidates: [{ id: 8120, dist: 0.742 }] },
      { hop: 2, nodeId: 14200, layer: 1, distance: 0.589, deltaDist: 0.153, isStagnant: false, candidates: [{ id: 14200, dist: 0.589 }] },
      { hop: 3, nodeId: 21050, layer: 0, distance: 0.460, deltaDist: 0.129, isStagnant: false, candidates: [{ id: 21050, dist: 0.460 }] },
      { hop: 4, nodeId: 28900, layer: 0, distance: 0.360, deltaDist: 0.100, isStagnant: false, candidates: [{ id: 28900, dist: 0.360 }] },
      { hop: 5, nodeId: 35400, layer: 0, distance: 0.280, deltaDist: 0.080, isStagnant: false, candidates: [{ id: 35400, dist: 0.280 }] },
      { hop: 6, nodeId: 41200, layer: 0, distance: 0.210, deltaDist: 0.070, isStagnant: false, candidates: [{ id: 41200, dist: 0.210 }] },
      { hop: 7, nodeId: 45600, layer: 0, distance: 0.160, deltaDist: 0.050, isStagnant: false, candidates: [{ id: 45600, dist: 0.160 }] },
      { hop: 8, nodeId: 49100, layer: 0, distance: 0.135, deltaDist: 0.025, isStagnant: false, candidates: [{ id: 49100, dist: 0.135 }] },
    ]
  },
  'q2': {
    name: 'Query #2 (Boundary Hub)',
    description: 'Inter-cluster boundary query navigating around hub nodes penalized by hubness regulation.',
    exitHop: 18,
    evals: 890.2,
    hops: [
      { hop: 0, nodeId: 9812, layer: 3, distance: 0.880, deltaDist: 0.000, isStagnant: false, candidates: [{ id: 9812, dist: 0.880 }] },
      { hop: 1, nodeId: 15400, layer: 2, distance: 0.690, deltaDist: 0.190, isStagnant: false, candidates: [{ id: 15400, dist: 0.690 }] },
      { hop: 2, nodeId: 23100, layer: 1, distance: 0.510, deltaDist: 0.180, isStagnant: false, candidates: [{ id: 23100, dist: 0.510 }] },
      { hop: 3, nodeId: 31200, layer: 0, distance: 0.370, deltaDist: 0.140, isStagnant: false, candidates: [{ id: 31200, dist: 0.370 }] },
      { hop: 4, nodeId: 39500, layer: 0, distance: 0.260, deltaDist: 0.110, isStagnant: false, candidates: [{ id: 39500, dist: 0.260 }] },
    ]
  }
};

export const QueryCanvas: React.FC = () => {
  const [selectedQueryKey, setSelectedQueryKey] = useState<'q0' | 'q1' | 'q2'>('q0');
  const [currentHopIdx, setCurrentHopIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const shouldReduceMotion = useReducedMotion();

  const activeQuery = QUERY_TRACES[selectedQueryKey];
  const activeHop = activeQuery.hops[currentHopIdx] || activeQuery.hops[0];

  // Playback timer
  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentHopIdx((prev) => {
          if (prev >= activeQuery.hops.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 350);
    }
    return () => clearInterval(timer);
  }, [isPlaying, activeQuery.hops.length]);

  return (
    <div className={styles.canvasContainer}>
      {/* Precision Workspace Bar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <div className={styles.segmentedGroup}>
            <button
              className={`${styles.segBtn} ${selectedQueryKey === 'q0' ? styles.segBtnActive : ''}`}
              onClick={() => { setSelectedQueryKey('q0'); setCurrentHopIdx(0); setIsPlaying(false); }}
            >
              Query #0 (Dense)
            </button>
            <button
              className={`${styles.segBtn} ${selectedQueryKey === 'q1' ? styles.segBtnActive : ''}`}
              onClick={() => { setSelectedQueryKey('q1'); setCurrentHopIdx(0); setIsPlaying(false); }}
            >
              Query #1 (Ridge)
            </button>
            <button
              className={`${styles.segBtn} ${selectedQueryKey === 'q2' ? styles.segBtnActive : ''}`}
              onClick={() => { setSelectedQueryKey('q2'); setCurrentHopIdx(0); setIsPlaying(false); }}
            >
              Query #2 (Boundary)
            </button>
          </div>

          <span className={styles.barSep}>/</span>
          <span className="badge-pill illustrative">
            <Activity size={10} />
            <span>ILLUSTRATIVE SIMULATION &bull; ADAPTIVE EF CONVERGENCE</span>
          </span>
        </div>

        <div className={styles.barRight}>
          <span className="text-xs text-muted tabular-nums">
            SIFT-100K Testbed &bull; Target: Hop {activeQuery.exitHop}
          </span>
        </div>
      </div>

      {/* Main Dual-Pane Stage */}
      <div className={styles.queryStageWrapper}>
        {/* Left Pane: Traversal Beam Search Sequence & Convergence Curve */}
        <div className={styles.queryLeftPane}>
          {/* Controls toolbar */}
          <div className={styles.queryTrajectoryControls}>
            <div className={styles.playbackGroup}>
              <button
                className={`${styles.playbackBtn} ${isPlaying ? styles.playbackBtnActive : ''}`}
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                <span>{isPlaying ? 'Pause' : 'Play Traversal'}</span>
              </button>

              <button
                className={styles.playbackBtn}
                onClick={() => setCurrentHopIdx((prev) => Math.min(prev + 1, activeQuery.hops.length - 1))}
                disabled={currentHopIdx >= activeQuery.hops.length - 1}
              >
                <SkipForward size={12} />
                <span>Step Hop</span>
              </button>

              <button
                className={styles.playbackBtn}
                onClick={() => { setCurrentHopIdx(0); setIsPlaying(false); }}
              >
                <RotateCcw size={12} />
                <span>Reset</span>
              </button>
            </div>

            <div className={styles.hopStepIndicator}>
              Hop <span className="text-accent-glow font-semibold">{activeHop.hop}</span> of {activeQuery.hops.length - 1} &bull; Node #{activeHop.nodeId}
            </div>
          </div>

          {/* Hop Track Carousel */}
          <div className={styles.hopSequenceDisplay}>
            <div className="text-xs text-muted font-mono" style={{ textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Beam Search Geodesic Steps
            </div>

            <div className={styles.hopTrack}>
              {activeQuery.hops.map((h, idx) => {
                const isCurrent = idx === currentHopIdx;
                const isExit = idx === activeQuery.exitHop;

                return (
                  <React.Fragment key={`hop-${h.hop}`}>
                    <div
                      className={`${styles.hopBadge} ${isCurrent ? styles.hopBadgeActive : h.isStagnant ? styles.hopBadgeStagnant : ''}`}
                      onClick={() => setCurrentHopIdx(idx)}
                    >
                      <span style={{ opacity: 0.7 }}>Hop {h.hop}</span>
                      <span className="font-semibold text-primary">#{h.nodeId}</span>
                      {isExit && <span style={{ fontSize: '8.5px', color: 'var(--semantic-emerald)' }}>EXIT</span>}
                    </div>
                    {idx < activeQuery.hops.length - 1 && <span className={styles.hopArrowSep}>&rarr;</span>}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          {/* SVG Convergence Curve: Distance Delta vs. Epsilon Threshold */}
          <div className={styles.convergenceGraphArea}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="text-xs font-semibold text-primary">
                Convergence Rate: &Delta;Distance per Search Hop
              </span>
              <span className="badge-pill emerald">
                Stagnation Threshold: &epsilon; = 10⁻⁴ (p=6)
              </span>
            </div>

            <svg viewBox="0 0 520 180" className={styles.convergenceSvg}>
              <defs>
                <linearGradient id="convergenceAreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#C97D4A" stopOpacity="0.22" />
                  <stop offset="85%" stopColor="#C97D4A" stopOpacity="0.04" />
                  <stop offset="100%" stopColor="#C97D4A" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Threshold Line at epsilon = 10^-4 */}
              <line x1={40} y1={140} x2={480} y2={140} stroke="var(--semantic-emerald)" strokeWidth={1} strokeDasharray="3 3" />
              <text x={485} y={143} fill="var(--semantic-emerald)" fontFamily="var(--font-mono)" fontSize="9">
                &epsilon;=10⁻⁴
              </text>

              {/* Shaded Stagnation Window (Hops 11 - 16) */}
              {selectedQueryKey === 'q0' && (
                <>
                  <rect x={315} y={20} width={135} height={120} fill="rgba(107, 163, 126, 0.1)" stroke="rgba(107, 163, 126, 0.25)" strokeDasharray="2 2" />
                  <text x={325} y={35} fill="var(--semantic-emerald)" fontFamily="var(--font-mono)" fontSize="9" fontWeight={600}>
                    STAGNATION WINDOW (p=6)
                  </text>
                </>
              )}

              {/* Area Gradient Fill under Distance Trajectory */}
              <polygon
                fill="url(#convergenceAreaGrad)"
                points={
                  activeQuery.hops.map((h, i) => {
                    const x = 50 + (i / (activeQuery.hops.length - 1)) * 400;
                    const y = 30 + (1 - (h.distance - 0.08) / 0.8) * 110;
                    return `${x},${Math.min(145, Math.max(30, y))}`;
                  }).join(' ') + ` ${50 + 400},140 50,140`
                }
              />

              {/* Graph Trajectory Line */}
              <polyline
                fill="none"
                stroke="var(--accent)"
                strokeWidth={2.2}
                points={activeQuery.hops.map((h, i) => {
                  const x = 50 + (i / (activeQuery.hops.length - 1)) * 400;
                  const y = 30 + (1 - (h.distance - 0.08) / 0.8) * 110;
                  return `${x},${Math.min(145, Math.max(30, y))}`;
                }).join(' ')}
              />

              {/* Data points with Active Hop Radar Ring & Clickable Targets */}
              {activeQuery.hops.map((h, i) => {
                const x = 50 + (i / (activeQuery.hops.length - 1)) * 400;
                const y = 30 + (1 - (h.distance - 0.08) / 0.8) * 110;
                const clampedY = Math.min(145, Math.max(30, y));
                const isSelected = i === currentHopIdx;

                return (
                  <g
                    key={`pt-${i}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setCurrentHopIdx(i)}
                  >
                    {/* Generous Hit Circle */}
                    <circle cx={x} cy={clampedY} r={14} fill="transparent" />

                    {/* Concentric Centered Active Radar Pulse */}
                    {isSelected && (
                      <circle
                        cx={x}
                        cy={clampedY}
                        r={12}
                        fill="none"
                        stroke="var(--accent-glow)"
                        strokeWidth={1.8}
                        strokeDasharray="2 2"
                        className={styles.radarPulseRing}
                        style={{ filter: 'drop-shadow(0 0 4px var(--accent-glow))' }}
                      />
                    )}

                    {/* Data Node Circle */}
                    <circle
                      cx={x}
                      cy={clampedY}
                      r={isSelected ? 5.5 : 3.5}
                      fill={isSelected ? 'var(--accent-glow)' : 'var(--bg-card)'}
                      stroke={isSelected ? '#ffffff' : 'var(--accent)'}
                      strokeWidth={isSelected ? 2 : 1.5}
                      style={{ transition: 'r 0.15s ease, fill 0.15s ease, stroke-width 0.15s ease' }}
                    />
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Right Pane: Candidate Buffer & Analytical Readout */}
        <div className={styles.queryRightPane}>
          <div className={styles.paneHeader}>
            <span className={styles.paneTitle}>Active Candidate Priority Queue (W Buffer)</span>
            <span className="badge-pill accent">Layer {activeHop.layer}</span>
          </div>

          <div className={styles.candidateBufferList}>
            <div className={styles.sectionNotice}>
              {activeQuery.description}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <span className="text-xs text-muted font-mono" style={{ textTransform: 'uppercase' }}>
                Top Nearest Neighbors at Hop {activeHop.hop}:
              </span>

              <AnimatePresence mode="popLayout">
                {activeHop.candidates.map((c, rank) => (
                  <motion.div
                    key={`cand-${c.id}`}
                    layout
                    initial={shouldReduceMotion ? false : { opacity: 0, y: -8, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={
                      shouldReduceMotion
                        ? { opacity: 0 }
                        : { opacity: 0, scale: 0.95, height: 0, marginTop: 0, marginBottom: 0, transition: { duration: 0.12 } }
                    }
                    transition={{ type: 'spring', stiffness: 450, damping: 32 }}
                    className={`${styles.candidateRow} ${rank === 0 ? styles.candidateRowTop : ''}`}
                  >
                    <span className={styles.candidateRank}>#{rank + 1}</span>
                    <span className={styles.candidateId}>Node #{c.id}</span>
                    <span className="tabular-nums text-xs text-muted">Distance:</span>
                    <span className={`${styles.candidateDist} tabular-nums`}>{c.dist.toFixed(5)}</span>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            {/* Empirical Speedup Readout */}
            <div className={styles.regimeBlock} style={{ marginTop: '12px' }}>
              <div className={styles.blockRow}>
                <span className={styles.blockBadge}>Step 5 Ada-ef Speedup</span>
                <span className="badge-pill emerald">+50.3% QPS</span>
              </div>
              <div className={styles.blockNumberRow}>
                <span className={`${styles.blockHeroNum} tabular-nums text-accent-glow`}>
                  {activeQuery.evals.toFixed(1)}
                </span>
                <span className={styles.blockHeroUnit}>Evals / Query</span>
              </div>
              <div className={styles.blockSub}>
                Terminates search at <strong className="text-primary">Hop {activeQuery.exitHop}</strong> vs. Hop 24 Baseline (&minus;30.1% distance evals evaluated on real Intel Core 5 210H).
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
