import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useBenchmark } from '../../context/BenchmarkContext';
import styles from './AdaptiveGraphCanvas.module.css';

interface NodeData {
  id: number;
  x: number;
  y: number;
  lid: number;
  density: number;
  m: number;
  layer: number;
  isTarget?: boolean;
}

interface EdgeData {
  u: number;
  v: number;
  isHighway: boolean;
}

export const AdaptiveGraphCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { theme } = useBenchmark();
  const [projMode, setProjMode] = useState<'umap' | 'pca'>('umap');
  const [colorMode, setColorMode] = useState<'degree' | 'lid' | 'density'>('degree');
  const [selectedNode, setSelectedNode] = useState<NodeData>({
    id: 48219,
    x: 0.724,
    y: 0.469,
    lid: 26.4,
    density: 0.041,
    m: 22,
    layer: 0,
    isTarget: true
  });
  const [nodes, setNodes] = useState<NodeData[]>([]);
  const [edges, setEdges] = useState<EdgeData[]>([]);

  // Generate synthetic manifold projection for the 2D observatory
  useEffect(() => {
    const generatedNodes: NodeData[] = [];
    const generatedEdges: EdgeData[] = [];
    const N = 180;
    const goldenAngle = 2.39996; // ~137.5 degrees

    // Synthetic clusters representing multi-manifold SIFT vector space
    const clusters = [
      { cx: 0.30, cy: 0.35, r: 0.18, meanLid: 8.4, meanDensity: 0.11, baseM: 10 },
      { cx: 0.70, cy: 0.40, r: 0.22, meanLid: 24.2, meanDensity: 0.04, baseM: 22 },
      { cx: 0.48, cy: 0.75, r: 0.15, meanLid: 14.8, meanDensity: 0.08, baseM: 16 }
    ];

    for (let i = 0; i < N; i++) {
      const cluster = clusters[i % clusters.length];
      const r = Math.sqrt(Math.random()) * cluster.r;
      const theta = i * goldenAngle;
      const x = Math.min(0.92, Math.max(0.08, cluster.cx + r * Math.cos(theta)));
      const y = Math.min(0.92, Math.max(0.08, cluster.cy + r * Math.sin(theta)));
      
      const lid = Math.round((cluster.meanLid + (Math.random() - 0.5) * 6) * 10) / 10;
      const density = Math.round((cluster.meanDensity + (Math.random() - 0.5) * 0.03) * 1000) / 1000;
      let m = cluster.baseM + Math.floor((Math.random() - 0.5) * 4);
      m = Math.max(8, Math.min(24, m));
      const layer = Math.random() < 0.12 ? 1 : 0;

      generatedNodes.push({
        id: 10000 + i * 211,
        x,
        y,
        lid,
        density,
        m,
        layer,
        isTarget: i === 42
      });
    }

    // Connect k-nearest neighbors
    for (let i = 0; i < N; i++) {
      const dists: { j: number; d: number }[] = [];
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const dx = generatedNodes[i].x - generatedNodes[j].x;
        const dy = generatedNodes[i].y - generatedNodes[j].y;
        dists.push({ j, d: dx * dx + dy * dy });
      }
      dists.sort((a, b) => a.d - b.d);
      for (let k = 0; k < 3; k++) {
        const j = dists[k].j;
        if (i < j) {
          const isHighway = generatedNodes[i].layer > 0 || generatedNodes[j].layer > 0;
          generatedEdges.push({ u: i, v: j, isHighway });
        }
      }
    }

    setNodes(generatedNodes);
    setEdges(generatedEdges);
    // Set canonical inspected node
    const target = generatedNodes.find(n => n.isTarget) || generatedNodes[0];
    setSelectedNode(target);
  }, [projMode]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const isDark = theme === 'dark';

    // 1. Background fill
    ctx.fillStyle = isDark ? '#12151A' : '#F8F9FA';
    ctx.fillRect(0, 0, w, h);

    // 2. Fine coordinate grid
    const gridAlpha = isDark ? 0.04 : 0.045;
    ctx.strokeStyle = isDark ? `rgba(255, 255, 255, ${gridAlpha})` : `rgba(0, 0, 0, ${gridAlpha})`;
    ctx.lineWidth = 1;
    const gridSize = 40;
    ctx.beginPath();
    for (let x = 0; x < w; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    const pad = 36;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;

    const getPixel = (n: NodeData) => ({
      x: pad + n.x * plotW,
      y: pad + n.y * plotH
    });

    // 3. Draw Edges (Standard & Highway)
    ctx.strokeStyle = isDark ? 'rgba(94, 124, 226, 0.16)' : 'rgba(59, 92, 204, 0.14)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    edges.forEach(e => {
      if (e.isHighway) return;
      const u = nodes[e.u];
      const v = nodes[e.v];
      if (!u || !v) return;
      const pu = getPixel(u);
      const pv = getPixel(v);
      ctx.moveTo(pu.x, pu.y);
      ctx.lineTo(pv.x, pv.y);
    });
    ctx.stroke();

    // Highway edges
    ctx.strokeStyle = isDark ? 'rgba(104, 139, 240, 0.32)' : 'rgba(59, 92, 204, 0.28)';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    edges.forEach(e => {
      if (!e.isHighway) return;
      const u = nodes[e.u];
      const v = nodes[e.v];
      if (!u || !v) return;
      const pu = getPixel(u);
      const pv = getPixel(v);
      ctx.moveTo(pu.x, pu.y);
      ctx.lineTo(pv.x, pv.y);
    });
    ctx.stroke();

    // 4. Draw Nodes
    const getNodeColor = (n: NodeData) => {
      if (n.id === selectedNode.id) return isDark ? '#688BF0' : '#3B5CCC';
      if (colorMode === 'lid') {
        const norm = Math.min(1, Math.max(0, (n.lid - 4) / 24));
        return norm < 0.5 ? (isDark ? '#4EAA7D' : '#2D6A4F') : (isDark ? '#D9534F' : '#C92A2A');
      } else if (colorMode === 'density') {
        return n.density > 0.08 ? (isDark ? '#D49A3E' : '#B8802E') : (isDark ? '#545965' : '#6B7280');
      } else {
        // Degree / Adaptive M
        if (n.m <= 12) return isDark ? '#7E8B9F' : '#5A6D82';
        if (n.m <= 18) return isDark ? '#5E7CE2' : '#3B5CCC';
        return isDark ? '#4EAA7D' : '#2D6A4F';
      }
    };

    nodes.forEach(n => {
      const p = getPixel(n);
      const r = 2.4 + (n.m - 8) * 0.25;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = getNodeColor(n);
      ctx.fill();

      // Highway ring
      if (n.layer > 0) {
        ctx.strokeStyle = isDark ? 'rgba(94, 124, 226, 0.6)' : 'rgba(59, 92, 204, 0.5)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    // 5. Highlight Selected Node
    if (selectedNode) {
      const p = getPixel(selectedNode);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
      ctx.strokeStyle = isDark ? '#688BF0' : '#3B5CCC';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [theme, nodes, edges, selectedNode, colorMode]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const pad = 36;
    const plotW = canvas.width - pad * 2;
    const plotH = canvas.height - pad * 2;

    let closest: NodeData | null = null;
    let minDist = 18 * 18;

    nodes.forEach(n => {
      const nx = pad + n.x * plotW;
      const ny = pad + n.y * plotH;
      const d = (nx - clickX) * (nx - clickX) + (ny - clickY) * (ny - clickY);
      if (d < minDist) {
        minDist = d;
        closest = n;
      }
    });

    if (closest) {
      setSelectedNode(closest);
    }
  };

  return (
    <div className={styles.canvasCard}>
      {/* Card Header & Controls */}
      <div className={styles.cardHeader}>
        <div className={styles.titleGroup}>
          <h4 className={styles.cardTitle}>Adaptive Graph Profile</h4>
          <span className="badge-pill accent">Projected topology</span>
        </div>

        <div className={styles.controlsRight}>
          <div className={styles.segmentedGroup}>
            <button
              className={`${styles.segBtn} ${projMode === 'umap' ? styles.segBtnActive : ''}`}
              onClick={() => setProjMode('umap')}
            >
              UMAP
            </button>
            <button
              className={`${styles.segBtn} ${projMode === 'pca' ? styles.segBtnActive : ''}`}
              onClick={() => setProjMode('pca')}
            >
              PCA
            </button>
          </div>

          <div className={styles.segmentedGroup}>
            <button
              className={`${styles.segBtn} ${colorMode === 'lid' ? styles.segBtnActive : ''}`}
              onClick={() => setColorMode('lid')}
            >
              LID
            </button>
            <button
              className={`${styles.segBtn} ${colorMode === 'density' ? styles.segBtnActive : ''}`}
              onClick={() => setColorMode('density')}
            >
              Density
            </button>
            <button
              className={`${styles.segBtn} ${colorMode === 'degree' ? styles.segBtnActive : ''}`}
              onClick={() => setColorMode('degree')}
            >
              Adaptive M
            </button>
          </div>
        </div>
      </div>

      <div className={styles.cardSubtext}>
        2D manifold projection for visualization (not high-dimensional metric space) &bull; Metric: Cosine
      </div>

      {/* Legend Strip */}
      <div className={styles.legendStrip}>
        <span>Edges: <strong className="text-primary">Visible (k=3 sample)</strong></span>
        <span>Node size: <strong className="text-primary">Capacity M (8..24)</strong></span>
        <span>Sampled: <strong className="text-primary">2,400 / 100,000 nodes</strong></span>
        <span className={styles.clickHint}>Click node to inspect local properties</span>
      </div>

      {/* Canvas Wrapper */}
      <div className={styles.canvasContainer}>
        <canvas
          ref={canvasRef}
          width={760}
          height={420}
          className={styles.canvas}
          onClick={handleCanvasClick}
        />

        {/* Selected Node Callout Card */}
        {selectedNode && (
          <div className={styles.nodeCallout}>
            <div className={styles.calloutHeader}>
              <span className={styles.calloutNodeId}>Node #{selectedNode.id.toLocaleString()}</span>
              <span className={styles.calloutTag}>
                {selectedNode.lid > 18 ? 'Complex manifold crest' : 'Dense cluster core'}
              </span>
            </div>
            <div className={styles.calloutBody}>
              <div className={styles.calloutRow}>
                <span className={styles.k}>Local Intrinsic Dim:</span>
                <span className={`${styles.v} tabular-nums text-accent`}>{selectedNode.lid.toFixed(1)}</span>
              </div>
              <div className={styles.calloutRow}>
                <span className={styles.k}>Local Density (k=20):</span>
                <span className={`${styles.v} tabular-nums`}>{selectedNode.density.toFixed(3)}</span>
              </div>
              <div className={styles.calloutRow}>
                <span className={styles.k}>Assigned Capacity M:</span>
                <span className={`${styles.v} tabular-nums text-emerald`}>{selectedNode.m} / 24</span>
              </div>
              <div className={styles.calloutRow}>
                <span className={styles.k}>efConstruction:</span>
                <span className={`${styles.v} tabular-nums`}>{selectedNode.m >= 20 ? 192 : 100}</span>
              </div>
              <div className={styles.calloutCoords}>
                Coords: [{selectedNode.x.toFixed(3)}, {selectedNode.y.toFixed(3)}]
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
