/* 2D Manifold Graph Projection & Streaming Churn / Localized Healing Simulator */
import React, { useState, useMemo } from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { Info, Trash2, Zap, RotateCcw } from 'lucide-react';
import styles from './CanvasShared.module.css';

interface NodePoint {
  id: number;
  label: string;
  umapX: number;
  umapY: number;
  pcaX: number;
  pcaY: number;
  lid: number;
  density: number;
  degreeM: number;
  efC: number;
  cluster: 'dense' | 'ridge' | 'bridge';
}

// Generate realistic manifold scatter of 90 sampled graph nodes
const GENERATED_NODES: NodePoint[] = Array.from({ length: 90 }, (_, i) => {
  const isDense = i < 35;
  const isRidge = i >= 35 && i < 70;
  const cluster = isDense ? 'dense' : isRidge ? 'ridge' : 'bridge';

  let umapX = 0, umapY = 0, pcaX = 0, pcaY = 0;
  let lid = 10, density = 0.05, degreeM = 16, efC = 100;

  if (isDense) {
    const angle = (i / 35) * Math.PI * 2;
    const r = 0.08 + ((i * 17) % 20) / 100;
    umapX = -0.38 + Math.cos(angle) * r;
    umapY = 0.22 + Math.sin(angle) * r;
    pcaX = -0.45 + Math.cos(angle) * (r * 1.3);
    pcaY = 0.15 + Math.sin(angle) * (r * 0.9);
    lid = 8.2 + ((i * 7) % 40) / 10;
    density = 0.08 + ((i * 11) % 50) / 1000;
    degreeM = 10;
    efC = 70;
  } else if (isRidge) {
    const t = (i - 35) / 35;
    const jitterX = (((i * 23) % 25) - 12) / 100;
    const jitterY = (((i * 31) % 25) - 12) / 100;
    umapX = 0.15 + t * 0.5 + jitterX;
    umapY = -0.05 - t * 0.45 + jitterY;
    pcaX = 0.1 + t * 0.6 + jitterX;
    pcaY = -0.1 - t * 0.35 + jitterY;
    lid = 22.0 + ((i * 13) % 94) / 10;
    density = 0.02 + ((i * 5) % 25) / 1000;
    degreeM = 22 + (i % 3 === 0 ? 2 : 0);
    efC = 192;
  } else {
    const t = (i - 70) / 20;
    umapX = -0.25 + t * 0.45 + (((i * 19) % 15) - 7) / 100;
    umapY = 0.15 - t * 0.25 + (((i * 29) % 15) - 7) / 100;
    pcaX = -0.3 + t * 0.5;
    pcaY = 0.05 - t * 0.2;
    lid = 14.5 + ((i * 11) % 50) / 10;
    density = 0.045;
    degreeM = 16;
    efC = 110;
  }

  return {
    id: i === 48 ? 48219 : 10000 + i * 382,
    label: i === 48 ? 'Node #48,219 (Anchor)' : `Node #${10000 + i * 382}`,
    umapX,
    umapY,
    pcaX,
    pcaY,
    lid,
    density,
    degreeM,
    efC,
    cluster,
  };
});

// Highway edges connecting adjacent nodes
const EDGES: [number, number][] = [
  ...Array.from({ length: 28 }, (_, i) => [i, (i + 1) % 35] as [number, number]),
  ...Array.from({ length: 15 }, (_, i) => [i, (i + 5) % 35] as [number, number]),
  ...Array.from({ length: 25 }, (_, i) => [35 + i, 35 + ((i + 1) % 35)] as [number, number]),
  ...Array.from({ length: 15 }, (_, i) => [35 + i, 35 + ((i + 4) % 35)] as [number, number]),
  [22, 70], [70, 75], [75, 80], [80, 85], [85, 48], [48, 52], [72, 30], [78, 42]
];

export const ManifoldCanvas: React.FC = () => {
  const { toggleDrawer } = useBenchmark();
  const [projection, setProjection] = useState<'umap' | 'pca'>('umap');
  const [colorMode, setColorMode] = useState<'degree' | 'lid' | 'density'>('degree');
  const [selectedNodeId, setSelectedNodeId] = useState<number>(48219);

  // Frontier 2 Interactive Churn & Healing Simulator State
  const [isChurnMode, setIsChurnMode] = useState<boolean>(false);
  const [deletedIndices, setDeletedIndices] = useState<Set<number>>(new Set());
  const [isCompacted, setIsCompacted] = useState<boolean>(false);
  const [compactionNotice, setCompactionNotice] = useState<string | null>(null);

  // Precomputed Adjacency Map for Localized Healing Calculations
  const adjacency = useMemo(() => {
    const adj = new Map<number, number[]>();
    for (let i = 0; i < GENERATED_NODES.length; i++) adj.set(i, []);
    for (const [u, v] of EDGES) {
      adj.get(u)?.push(v);
      adj.get(v)?.push(u);
    }
    return adj;
  }, []);

  // Compute surviving base edges
  const survivingEdges = useMemo(() => {
    return EDGES.filter(([u, v]) => !deletedIndices.has(u) && !deletedIndices.has(v));
  }, [deletedIndices]);

  // Compute soft transit ghost edges (edges where either endpoint is deleted)
  const ghostTransitEdges = useMemo(() => {
    if (isCompacted) return [];
    return EDGES.filter(([u, v]) => deletedIndices.has(u) || deletedIndices.has(v));
  }, [deletedIndices, isCompacted]);

  // Compute Frontier 2 Localized Graph Healing edges:
  // For every deleted node v, rewire predecessor u to surviving neighbors w
  const healedEdges = useMemo(() => {
    if (isCompacted || deletedIndices.size === 0) return [];
    const healed: [number, number][] = [];
    const added = new Set<string>();

    deletedIndices.forEach((deadIdx) => {
      const neighbors = (adjacency.get(deadIdx) || []).filter((nbr) => !deletedIndices.has(nbr));
      for (let i = 0; i < neighbors.length; i++) {
        for (let j = i + 1; j < neighbors.length; j++) {
          const u = Math.min(neighbors[i], neighbors[j]);
          const v = Math.max(neighbors[i], neighbors[j]);
          const key = `${u}-${v}`;
          if (!added.has(key)) {
            added.add(key);
            healed.push([u, v]);
          }
        }
      }
    });

    return healed;
  }, [deletedIndices, isCompacted, adjacency]);

  // Nodes to display (if compacted, deleted nodes vanish entirely)
  const renderedNodes = useMemo(() => {
    return GENERATED_NODES.map((node, idx) => ({
      ...node,
      originalIndex: idx,
      isDeleted: deletedIndices.has(idx),
    })).filter((node) => !isCompacted || !node.isDeleted);
  }, [deletedIndices, isCompacted]);

  const selectedNode = useMemo(() => {
    return GENERATED_NODES.find((n) => n.id === selectedNodeId) || GENERATED_NODES[48];
  }, [selectedNodeId]);

  // SVG coordinate transformation
  const toSvgCoords = (node: { umapX: number; umapY: number; pcaX: number; pcaY: number }) => {
    const x = projection === 'umap' ? node.umapX : node.pcaX;
    const y = projection === 'umap' ? node.umapY : node.pcaY;
    const svgX = 400 + x * 420;
    const svgY = 240 - y * 340;
    return { x: svgX, y: svgY };
  };

  const getNodeColor = (node: NodePoint, isDeleted: boolean) => {
    if (isDeleted) return 'rgba(239, 68, 68, 0.4)';
    if (colorMode === 'degree') {
      return node.degreeM >= 22 ? 'var(--accent-glow)' : node.degreeM === 16 ? 'var(--accent)' : 'var(--semantic-emerald)';
    } else if (colorMode === 'lid') {
      return node.lid > 20 ? 'var(--accent-glow)' : node.lid > 14 ? 'var(--semantic-amber)' : 'var(--semantic-emerald)';
    } else {
      return node.density > 0.07 ? 'var(--semantic-emerald)' : node.density > 0.04 ? 'var(--semantic-amber)' : 'var(--accent-glow)';
    }
  };

  // Node deletion click handler in Churn Mode
  const handleNodeClick = (node: NodePoint, idx: number) => {
    setSelectedNodeId(node.id);
    if (!isChurnMode) {
      toggleDrawer('inspector');
      return;
    }

    setDeletedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
    setIsCompacted(false);
  };

  // Batch Churn Trigger (20% or 50%)
  const triggerBatchChurn = (targetPct: number) => {
    const count = Math.floor(GENERATED_NODES.length * (targetPct / 100));
    const newDeleted = new Set<number>();
    // Exclude anchor node 48 from automatic deletion
    for (let i = 0; i < GENERATED_NODES.length; i++) {
      if (i !== 48 && (i * 17) % GENERATED_NODES.length < count) {
        newDeleted.add(i);
        if (newDeleted.size >= count) break;
      }
    }
    setDeletedIndices(newDeleted);
    setIsCompacted(false);
    setCompactionNotice(null);
  };

  // Online Zero-Downtime Compaction Trigger
  const triggerCompaction = () => {
    if (deletedIndices.size === 0) return;
    setIsCompacted(true);
    setCompactionNotice(
      `Compacted in 62.1 ms • Reclaimed 50.3% index RAM • Purged ${deletedIndices.size} dead nodes • 46.0× faster than rebuilding`
    );
  };

  // Reset Graph to Clean State
  const resetGraph = () => {
    setDeletedIndices(new Set());
    setIsCompacted(false);
    setCompactionNotice(null);
  };

  return (
    <div className={styles.canvasContainer}>
      {/* Precision Workspace Bar */}
      <div className={styles.workspaceBar}>
        <div className={styles.barLeft}>
          <div className={styles.segmentedGroup}>
            <button
              className={`${styles.segBtn} ${projection === 'umap' ? styles.segBtnActive : ''}`}
              onClick={() => setProjection('umap')}
            >
              UMAP Projection
            </button>
            <button
              className={`${styles.segBtn} ${projection === 'pca' ? styles.segBtnActive : ''}`}
              onClick={() => setProjection('pca')}
            >
              PCA Projection
            </button>
          </div>

          <span className={styles.barSep}>|</span>

          <div className={styles.segmentedGroup}>
            <button
              className={`${styles.segBtn} ${colorMode === 'degree' ? styles.segBtnActive : ''}`}
              onClick={() => setColorMode('degree')}
            >
              Adaptive M(x)
            </button>
            <button
              className={`${styles.segBtn} ${colorMode === 'lid' ? styles.segBtnActive : ''}`}
              onClick={() => setColorMode('lid')}
            >
              Local Intrinsic Dim (LID)
            </button>
            <button
              className={`${styles.segBtn} ${colorMode === 'density' ? styles.segBtnActive : ''}`}
              onClick={() => setColorMode('density')}
            >
              k-NN Density
            </button>
          </div>

          <span className={styles.barSep}>|</span>

          {/* Interactive Churn Simulator Toggle */}
          <button
            className={`${styles.segBtn} ${isChurnMode ? styles.segBtnActive : ''}`}
            onClick={() => setIsChurnMode(!isChurnMode)}
            style={{ color: isChurnMode ? 'var(--accent-glow)' : 'var(--text-muted)' }}
          >
            <Trash2 size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
            {isChurnMode ? 'Churn Simulator (Active)' : 'Enable Churn Simulator'}
          </button>
        </div>

        <div className={styles.barRight}>
          {isChurnMode && (
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <button
                className={styles.barActionBtn}
                onClick={() => triggerBatchChurn(20)}
                title="Simulate 20% streaming node deletion"
              >
                <span>Churn 20%</span>
              </button>
              <button
                className={styles.barActionBtn}
                onClick={() => triggerBatchChurn(50)}
                title="Simulate 50% streaming node deletion"
              >
                <span>Churn 50%</span>
              </button>
              <button
                className={styles.barActionBtn}
                onClick={triggerCompaction}
                disabled={deletedIndices.size === 0 || isCompacted}
                style={{
                  borderColor: isCompacted ? 'var(--border-subtle)' : 'var(--semantic-emerald)',
                  color: isCompacted ? 'var(--text-dim)' : 'var(--semantic-emerald)',
                }}
                title="Run zero-downtime compaction: compact() in 62.1 ms"
              >
                <Zap size={12} />
                <span>Compact (62.1ms)</span>
              </button>
              <button
                className={styles.barActionBtn}
                onClick={resetGraph}
                title="Reset graph to 100% active state"
              >
                <RotateCcw size={12} />
                <span>Reset</span>
              </button>
            </div>
          )}

          <button
            className={styles.barActionBtn}
            onClick={() => toggleDrawer('inspector')}
            title="Inspect Selected Node Telemetry (Cmd+I)"
          >
            <Info size={13} />
            <span>Inspector</span>
            <kbd className={styles.kbd}>Cmd+I</kbd>
          </button>
        </div>
      </div>

      {/* Main Observatory Stage */}
      <div className={styles.manifoldContainer}>
        {/* Top coordinate telemetry strip */}
        <div className={styles.manifoldHeaderStrip}>
          <span>PROJECTION: 2D {projection.toUpperCase()} OBSERVATORY • SCALE: [-1.0, +1.0]</span>
          <span>
            {isChurnMode
              ? `STREAMING CHURN: ${deletedIndices.size} DELETED (${((deletedIndices.size / 90) * 100).toFixed(0)}%) • HEALED EDGES: ${healedEdges.length} • ACTIVE: ${90 - deletedIndices.size}`
              : `SELECTED: ${selectedNode.label} • LID=${selectedNode.lid.toFixed(1)} • M=${selectedNode.degreeM}`}
          </span>
          <span>
            {isCompacted ? 'COMPACTED CONTIGUOUS BUFFERS [0, N_ACTIVE-1]' : isChurnMode ? 'CLICK ANY NODE TO DELETE/RESTORE' : 'GROUND PLANE: LAYER 0 HNSW'}
          </span>
        </div>

        <div className={styles.manifoldStage}>
          {/* Top Horizontal Calibrated Scale Ruler */}
          <div className={styles.manifoldRulerX}>
            <span>-1.0</span>
            <span>-0.8</span>
            <span>-0.6</span>
            <span>-0.4</span>
            <span>-0.2</span>
            <span>0.0</span>
            <span>+0.2</span>
            <span>+0.4</span>
            <span>+0.6</span>
            <span>+0.8</span>
            <span>+1.0</span>
          </div>

          {/* Left Vertical Calibrated Scale Ruler */}
          <div className={styles.manifoldRulerY}>
            <span>+1.0</span>
            <span>+0.6</span>
            <span>+0.2</span>
            <span>0.0</span>
            <span>-0.2</span>
            <span>-0.6</span>
            <span>-1.0</span>
          </div>

          {/* SVG Coordinate Observatory */}
          <div className={styles.manifoldSvgArea}>
            <svg
              viewBox="0 0 800 480"
              className={styles.manifoldSvg}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Origin Crosshair Reticle */}
              <line x1={400} y1={0} x2={400} y2={480} stroke="var(--border-subtle)" strokeWidth={1} strokeDasharray="3 3" />
              <line x1={0} y1={240} x2={800} y2={240} stroke="var(--border-subtle)" strokeWidth={1} strokeDasharray="3 3" />

              {/* Cluster Background Shading Envelopes */}
              <ellipse
                cx={toSvgCoords(GENERATED_NODES[0]).x}
                cy={toSvgCoords(GENERATED_NODES[0]).y}
                rx={110}
                ry={75}
                fill="rgba(107, 163, 126, 0.05)"
                stroke="rgba(107, 163, 126, 0.18)"
                strokeDasharray="2 3"
              />
              <text
                x={toSvgCoords(GENERATED_NODES[0]).x - 80}
                y={toSvgCoords(GENERATED_NODES[0]).y - 60}
                fill="var(--text-dim)"
                fontFamily="var(--font-mono)"
                fontSize="9"
              >
                DENSE CORE CLUSTER (M=10)
              </text>

              <ellipse
                cx={toSvgCoords(GENERATED_NODES[48]).x}
                cy={toSvgCoords(GENERATED_NODES[48]).y}
                rx={130}
                ry={90}
                fill="rgba(201, 125, 74, 0.05)"
                stroke="rgba(201, 125, 74, 0.18)"
                strokeDasharray="2 3"
              />
              <text
                x={toSvgCoords(GENERATED_NODES[48]).x - 60}
                y={toSvgCoords(GENERATED_NODES[48]).y + 75}
                fill="var(--text-dim)"
                fontFamily="var(--font-mono)"
                fontSize="9"
              >
                HIGH-LID RIDGE CLUSTER (M=22-24)
              </text>

              {/* 1. Surviving Base Edges */}
              {survivingEdges.map(([srcIdx, dstIdx], edgeIdx) => {
                const src = GENERATED_NODES[srcIdx];
                const dst = GENERATED_NODES[dstIdx];
                if (!src || !dst) return null;
                const p1 = toSvgCoords(src);
                const p2 = toSvgCoords(dst);
                const isHighlight = src.id === selectedNodeId || dst.id === selectedNodeId;

                return (
                  <line
                    key={`surv-edge-${edgeIdx}`}
                    x1={p1.x}
                    y1={p1.y}
                    x2={p2.x}
                    y2={p2.y}
                    className={isHighlight ? styles.graphEdgeHighlight : styles.graphEdge}
                    style={{ transition: 'all 0.2s ease' }}
                  />
                );
              })}

              {/* 2. Soft Transit Ghost Edges (Red Dashed) */}
              {ghostTransitEdges.map(([srcIdx, dstIdx], edgeIdx) => {
                const src = GENERATED_NODES[srcIdx];
                const dst = GENERATED_NODES[dstIdx];
                if (!src || !dst) return null;
                const p1 = toSvgCoords(src);
                const p2 = toSvgCoords(dst);

                return (
                  <line
                    key={`ghost-edge-${edgeIdx}`}
                    x1={p1.x}
                    y1={p1.y}
                    x2={p2.x}
                    y2={p2.y}
                    stroke="rgba(239, 68, 68, 0.35)"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                    style={{ transition: 'opacity 0.2s ease' }}
                  />
                );
              })}

              {/* 3. Frontier 2 Localized Healed Edges (Glowing Green #10b981) */}
              {healedEdges.map(([uIdx, vIdx], edgeIdx) => {
                const u = GENERATED_NODES[uIdx];
                const v = GENERATED_NODES[vIdx];
                if (!u || !v) return null;
                const p1 = toSvgCoords(u);
                const p2 = toSvgCoords(v);

                return (
                  <g key={`healed-edge-${edgeIdx}`}>
                    <line
                      x1={p1.x}
                      y1={p1.y}
                      x2={p2.x}
                      y2={p2.y}
                      stroke="#10b981"
                      strokeWidth={2}
                      opacity={0.85}
                      style={{ transition: 'all 0.25s ease' }}
                    />
                  </g>
                );
              })}

              {/* Reticle Guides to Axes for Selected Node */}
              {selectedNode && (
                <g pointerEvents="none">
                  <line
                    x1={toSvgCoords(selectedNode).x}
                    y1={0}
                    x2={toSvgCoords(selectedNode).x}
                    y2={480}
                    stroke="var(--accent)"
                    strokeWidth={0.8}
                    strokeDasharray="2 2"
                    opacity={0.5}
                  />
                  <line
                    x1={0}
                    y1={toSvgCoords(selectedNode).y}
                    x2={800}
                    y2={toSvgCoords(selectedNode).y}
                    stroke="var(--accent)"
                    strokeWidth={0.8}
                    strokeDasharray="2 2"
                    opacity={0.5}
                  />
                </g>
              )}

              {/* Graph Scatter Nodes */}
              {renderedNodes.map((node) => {
                const pos = toSvgCoords(node);
                const isSelected = node.id === selectedNodeId;
                const isDeleted = node.isDeleted;

                return (
                  <g
                    key={node.id}
                    className={styles.manifoldNode}
                    onClick={() => handleNodeClick(node, node.originalIndex)}
                  >
                    {/* Generous Hit Target Circle */}
                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={16}
                      fill="transparent"
                      style={{ cursor: 'pointer' }}
                    />

                    {/* Outer Selection Reticle Ring */}
                    {isSelected && (
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={12}
                        className={styles.nodeSelectedRing}
                      />
                    )}

                    {/* Tombstone Marker Ring (if deleted) */}
                    {isDeleted && (
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={9}
                        fill="transparent"
                        stroke="rgba(239, 68, 68, 0.85)"
                        strokeWidth={1.5}
                        strokeDasharray="2 2"
                      />
                    )}

                    {/* Node Dot */}
                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={isSelected ? 6 : isDeleted ? 3 : node.degreeM >= 22 ? 5 : 4}
                      fill={getNodeColor(node, isDeleted)}
                      stroke={isSelected ? '#ffffff' : isDeleted ? 'rgba(239, 68, 68, 0.9)' : 'var(--border-emphasis)'}
                      strokeWidth={isSelected ? 2 : 1}
                      className={styles.manifoldNodeCircle}
                    />
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Persistent Bottom Observatory HUD */}
        <div className={styles.manifoldFooterHud}>
          <div className={styles.manifoldHudGroup}>
            {isChurnMode ? (
              <>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Active Nodes:</span>
                  <span className="tabular-nums font-semibold text-emerald">{90 - deletedIndices.size} / 90</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Tombstones:</span>
                  <span className="tabular-nums font-semibold text-accent-glow">{deletedIndices.size}</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Healed Edges:</span>
                  <span className="tabular-nums font-semibold text-emerald">{healedEdges.length} links</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Compaction Time:</span>
                  <span className="tabular-nums text-muted">{isCompacted ? '62.1 ms (46.0×)' : 'Pending'}</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>RAM Reclaimed:</span>
                  <span className="tabular-nums text-emerald">{isCompacted ? '-50.3%' : '0.0%'}</span>
                </div>
              </>
            ) : (
              <>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Active Node:</span>
                  <span className={`${styles.hudChipValue} text-accent-glow`}>{selectedNode.label}</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Local Intrinsic Dim (MLE):</span>
                  <span className="tabular-nums font-semibold">{selectedNode.lid.toFixed(1)}</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>Assigned Degree M(x):</span>
                  <span className="tabular-nums font-semibold text-emerald">M = {selectedNode.degreeM}</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>k-NN Density:</span>
                  <span className="tabular-nums">{selectedNode.density.toFixed(3)}</span>
                </div>
                <div className={styles.hudChip}>
                  <span className={styles.hudChipLabel}>efConstruction:</span>
                  <span className="tabular-nums">{selectedNode.efC}</span>
                </div>
              </>
            )}
          </div>

          <div className="badge-pill illustrative">
            {compactionNotice ? (
              <span style={{ color: 'var(--semantic-emerald)' }}>{compactionNotice}</span>
            ) : isChurnMode ? (
              <span>Frontier 2 Simulator &bull; Localized Graph Healing</span>
            ) : (
              <span>Graph Telemetry &bull; 90 Rendered Anchors</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
