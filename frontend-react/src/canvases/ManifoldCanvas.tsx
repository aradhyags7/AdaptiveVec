import React, { useState, useMemo } from 'react';
import { useBenchmark } from '../context/BenchmarkContext';
import { Info } from 'lucide-react';
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
  // Cluster distribution
  const isDense = i < 35;
  const isRidge = i >= 35 && i < 70;
  const cluster = isDense ? 'dense' : isRidge ? 'ridge' : 'bridge';

  // Base coordinates
  let umapX = 0, umapY = 0, pcaX = 0, pcaY = 0;
  let lid = 10, density = 0.05, degreeM = 16, efC = 100;

  if (isDense) {
    // Dense cluster centered around (-0.35, 0.25)
    const angle = (i / 35) * Math.PI * 2;
    const r = 0.08 + ((i * 17) % 20) / 100;
    umapX = -0.38 + Math.cos(angle) * r;
    umapY = 0.22 + Math.sin(angle) * r;
    pcaX = -0.45 + Math.cos(angle) * (r * 1.3);
    pcaY = 0.15 + Math.sin(angle) * (r * 0.9);
    lid = 8.2 + ((i * 7) % 40) / 10; // low LID: 8.2 - 12.2
    density = 0.08 + ((i * 11) % 50) / 1000;
    degreeM = 10;
    efC = 70;
  } else if (isRidge) {
    // Sparse High-LID ridge centered around (+0.35, -0.22)
    const t = (i - 35) / 35;
    const jitterX = (((i * 23) % 25) - 12) / 100;
    const jitterY = (((i * 31) % 25) - 12) / 100;
    umapX = 0.15 + t * 0.5 + jitterX;
    umapY = -0.05 - t * 0.45 + jitterY;
    pcaX = 0.1 + t * 0.6 + jitterX;
    pcaY = -0.1 - t * 0.35 + jitterY;
    lid = 22.0 + ((i * 13) % 94) / 10; // high LID: 22.0 - 31.4
    density = 0.02 + ((i * 5) % 25) / 1000;
    degreeM = 22 + (i % 3 === 0 ? 2 : 0); // expanded M = 22 or 24
    efC = 192;
  } else {
    // Highway bridge connecting both clusters
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
  // Dense cluster internal edges
  ...Array.from({ length: 28 }, (_, i) => [i, (i + 1) % 35] as [number, number]),
  ...Array.from({ length: 15 }, (_, i) => [i, (i + 5) % 35] as [number, number]),
  // Ridge cluster internal edges
  ...Array.from({ length: 25 }, (_, i) => [35 + i, 35 + ((i + 1) % 35)] as [number, number]),
  ...Array.from({ length: 15 }, (_, i) => [35 + i, 35 + ((i + 4) % 35)] as [number, number]),
  // Bridge edges
  [22, 70], [70, 75], [75, 80], [80, 85], [85, 48], [48, 52], [72, 30], [78, 42]
];

export const ManifoldCanvas: React.FC = () => {
  const { toggleDrawer } = useBenchmark();
  const [projection, setProjection] = useState<'umap' | 'pca'>('umap');
  const [colorMode, setColorMode] = useState<'degree' | 'lid' | 'density'>('degree');
  const [selectedNodeId, setSelectedNodeId] = useState<number>(48219); // Node #48,219 by default

  const selectedNode = useMemo(() => {
    return GENERATED_NODES.find((n) => n.id === selectedNodeId) || GENERATED_NODES[48];
  }, [selectedNodeId]);

  // SVG viewBox coordinates: 800 x 480, centered around (0, 0)
  const toSvgCoords = (node: NodePoint) => {
    const x = projection === 'umap' ? node.umapX : node.pcaX;
    const y = projection === 'umap' ? node.umapY : node.pcaY;
    // Map [-0.8, +0.8] to [60, 740] and [-0.6, +0.6] to [40, 440]
    const svgX = 400 + x * 420;
    const svgY = 240 - y * 340;
    return { x: svgX, y: svgY };
  };

  const getNodeColor = (node: NodePoint) => {
    if (colorMode === 'degree') {
      return node.degreeM >= 22 ? 'var(--accent-glow)' : node.degreeM === 16 ? 'var(--accent)' : 'var(--semantic-emerald)';
    } else if (colorMode === 'lid') {
      return node.lid > 20 ? 'var(--accent-glow)' : node.lid > 14 ? 'var(--semantic-amber)' : 'var(--semantic-emerald)';
    } else {
      return node.density > 0.07 ? 'var(--semantic-emerald)' : node.density > 0.04 ? 'var(--semantic-amber)' : 'var(--accent-glow)';
    }
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

          <span className={styles.barSep}>/</span>
          <span className="badge-pill accent">Metric: Cosine (SIFT-100K 128-D)</span>
        </div>

        <div className={styles.barRight}>
          <button
            className={styles.barActionBtn}
            onClick={() => toggleDrawer('inspector')}
            title="Inspect Selected Node Telemetry (Cmd+I)"
          >
            <Info size={13} />
            <span>Node Inspector</span>
            <kbd className={styles.kbd}>Cmd+I</kbd>
          </button>
        </div>
      </div>

      {/* Main Observatory Stage */}
      <div className={styles.manifoldContainer}>
        {/* Top coordinate telemetry strip */}
        <div className={styles.manifoldHeaderStrip}>
          <span>PROJECTION: 2D {projection.toUpperCase()} COORDINATE OBSERVATORY • SCALE: [-1.0, +1.0]</span>
          <span>SELECTED: {selectedNode.label} • LID={selectedNode.lid.toFixed(1)} • M={selectedNode.degreeM}</span>
          <span>GROUND PLANE: LAYER 0 PROXIMITY GRAPH</span>
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

              {/* Proximity Graph Edges with Dynamic Endpoint Tracking */}
              {EDGES.map(([srcIdx, dstIdx], edgeIdx) => {
                const src = GENERATED_NODES[srcIdx];
                const dst = GENERATED_NODES[dstIdx];
                if (!src || !dst) return null;
                const p1 = toSvgCoords(src);
                const p2 = toSvgCoords(dst);
                const isHighlight = src.id === selectedNodeId || dst.id === selectedNodeId;

                return (
                  <line
                    key={`edge-${edgeIdx}`}
                    x1={p1.x}
                    y1={p1.y}
                    x2={p2.x}
                    y2={p2.y}
                    className={isHighlight ? styles.graphEdgeHighlight : styles.graphEdge}
                    style={{ transition: 'stroke 0.18s ease, stroke-width 0.18s ease, opacity 0.18s ease' }}
                  />
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
                    style={{ transition: 'x1 0.12s ease-out, x2 0.12s ease-out' }}
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
                    style={{ transition: 'y1 0.12s ease-out, y2 0.12s ease-out' }}
                  />
                </g>
              )}

              {/* Graph Scatter Nodes with Generous Hit Area & Phosphor Recalibration */}
              {GENERATED_NODES.map((node) => {
                const pos = toSvgCoords(node);
                const isSelected = node.id === selectedNodeId;

                return (
                  <g
                    key={node.id}
                    className={styles.manifoldNode}
                    onClick={() => {
                      setSelectedNodeId(node.id);
                      toggleDrawer('inspector');
                    }}
                  >
                    {/* Generous Invisible Hit Target Circle (16px radius for forgiving clicks) */}
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

                    {/* Node Dot with Phosphor Color Transition */}
                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={isSelected ? 6 : node.degreeM >= 22 ? 5 : 4}
                      fill={getNodeColor(node)}
                      stroke={isSelected ? '#ffffff' : 'var(--border-emphasis)'}
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
          </div>

          <div className="badge-pill illustrative">
            <span>Graph Telemetry • 90 Rendered Anchors</span>
          </div>
        </div>
      </div>
    </div>
  );
};
