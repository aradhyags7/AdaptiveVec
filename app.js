/**
 * AdaptiveVec Interactive Studio & Vector Index Telemetry Engine
 * Obsidian Dark Research Laboratory Interface
 * Matches Stitch Specification: Overview, Graph Explorer, Query Lab, Benchmarks
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================================================
  // Application State
  // ==========================================================================
  const state = {
    currentView: "view-overview",
    projectionMode: "umap", // "umap" | "pca"
    colorMode: "degree",    // "lid" | "density" | "degree"
    activeLayer: 0,
    targetNodeId: 48219,
    searchAlgo: "adaptive",
    kNeighbors: 10,
    efSearch: 64,
    audioEnabled: false,
    audioCtx: null,
    isSearching: false,
    
    // Overview Canvas State
    overview: {
      panX: 0,
      panY: 0,
      zoom: 1,
      isDragging: false,
      dragStartX: 0,
      dragStartY: 0,
      hoveredNode: null,
      selectedNode: 48219,
      nodes: [],
      edges: []
    },

    // Graph Explorer State
    explorer: {
      panX: 0,
      panY: 0,
      zoom: 1,
      isDragging: false,
      dragStartX: 0,
      dragStartY: 0,
      nodes: [],
      hops: [
        { id: 10402, layer: 4, dist: 0.8920, x: -160, y: -120, label: "ENTRY #10,402 (Layer 4)" },
        { id: 12890, layer: 3, dist: 0.6540, x: -90, y: -70, label: "HOP 1 (#12,890)" },
        { id: 34011, layer: 1, dist: 0.3812, x: -40, y: -20, label: "HOP 2 (#34,011)" },
        { id: 48219, layer: 0, dist: 0.0418, x: 10, y: 15, label: "TARGET #48,219 (Found)" }
      ]
    },

    // Query Lab State
    query: {
      trajectory: [
        { hop: 0, adaptiveDist: 0.892, baselineDist: 0.892 },
        { hop: 2, adaptiveDist: 0.720, baselineDist: 0.780 },
        { hop: 4, adaptiveDist: 0.540, baselineDist: 0.660 },
        { hop: 6, adaptiveDist: 0.380, baselineDist: 0.540 },
        { hop: 8, adaptiveDist: 0.220, baselineDist: 0.420 },
        { hop: 10, adaptiveDist: 0.120, baselineDist: 0.310 },
        { hop: 12, adaptiveDist: 0.065, baselineDist: 0.220 },
        { hop: 14, adaptiveDist: 0.045, baselineDist: 0.150 },
        { hop: 16, adaptiveDist: 0.0418, baselineDist: 0.095 }, // Early Exit
        { hop: 18, adaptiveDist: 0.0418, baselineDist: 0.065 },
        { hop: 20, adaptiveDist: 0.0418, baselineDist: 0.048 },
        { hop: 22, adaptiveDist: 0.0418, baselineDist: 0.042 },
        { hop: 24, adaptiveDist: 0.0418, baselineDist: 0.0418 }
      ]
    }
  };

  // ==========================================================================
  // Web Audio Synthesizer (Micro-Haptics)
  // ==========================================================================
  function initAudio() {
    if (!state.audioCtx && (window.AudioContext || window.webkitAudioContext)) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
  }

  function playHapticBeep(freq = 840, duration = 0.04, type = "sine", gainVal = 0.05) {
    if (!state.audioEnabled) return;
    initAudio();
    if (!state.audioCtx) return;
    try {
      const osc = state.audioCtx.createOscillator();
      const gain = state.audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, state.audioCtx.currentTime);
      gain.gain.setValueAtTime(gainVal, state.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(state.audioCtx.destination);
      osc.start();
      osc.stop(state.audioCtx.currentTime + duration);
    } catch (e) {
      // Ignore audio failure
    }
  }

  // ==========================================================================
  // VIEW NAVIGATION (Sidebar Navigation & Tabs)
  // ==========================================================================
  const navItems = document.querySelectorAll(".nav-item");
  const viewPanels = document.querySelectorAll(".view-panel");

  function switchView(targetViewId) {
    playHapticBeep(640, 0.03);
    navItems.forEach(btn => {
      const isMatch = btn.getAttribute("data-view") === targetViewId;
      btn.classList.toggle("active", isMatch);
    });

    viewPanels.forEach(panel => {
      panel.classList.toggle("active", panel.id === targetViewId);
    });

    state.currentView = targetViewId;

    // Trigger canvas resizing & re-rendering when view appears
    requestAnimationFrame(() => {
      if (targetViewId === "view-overview") {
        renderOverviewCanvas();
        drawDensityCurve();
      } else if (targetViewId === "view-graph-explorer") {
        renderGeodesicCanvas();
      } else if (targetViewId === "view-query-lab") {
        renderTrajectoryChart();
      } else if (targetViewId === "view-benchmarks") {
        renderBenchmarkCharts();
      }
    });
  }

  navItems.forEach(btn => {
    btn.addEventListener("click", () => {
      const viewId = btn.getAttribute("data-view");
      if (viewId) switchView(viewId);
    });
  });

  // Jump from Overview Callout Card to Graph Explorer
  const btnJumpSubgraph = document.getElementById("btn-jump-subgraph");
  if (btnJumpSubgraph) {
    btnJumpSubgraph.addEventListener("click", () => {
      switchView("view-graph-explorer");
    });
  }

  // Audio Toggle
  const btnAudioToggle = document.getElementById("btn-audio-toggle");
  const iconAudioOn = document.getElementById("icon-audio-on");
  const iconAudioOff = document.getElementById("icon-audio-off");
  if (btnAudioToggle) {
    btnAudioToggle.addEventListener("click", () => {
      state.audioEnabled = !state.audioEnabled;
      if (iconAudioOn && iconAudioOff) {
        iconAudioOn.classList.toggle("hidden", !state.audioEnabled);
        iconAudioOff.classList.toggle("hidden", state.audioEnabled);
      }
      if (state.audioEnabled) playHapticBeep(880, 0.06);
    });
  }

  // ==========================================================================
  // COMMAND PALETTE (Ctrl+K)
  // ==========================================================================
  const searchModal = document.getElementById("search-modal");
  const btnOpenSearch = document.getElementById("btn-open-search");
  const cmdInput = document.getElementById("cmd-palette-input");
  const cmdItems = document.querySelectorAll(".cmd-item");

  function openCommandPalette() {
    if (searchModal) {
      searchModal.classList.remove("hidden");
      if (cmdInput) {
        cmdInput.value = "";
        cmdInput.focus();
      }
    }
  }

  function closeCommandPalette() {
    if (searchModal) {
      searchModal.classList.add("hidden");
    }
  }

  if (btnOpenSearch) {
    btnOpenSearch.addEventListener("click", openCommandPalette);
  }

  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (searchModal && searchModal.classList.contains("hidden")) {
        openCommandPalette();
      } else {
        closeCommandPalette();
      }
    } else if (e.key === "Escape") {
      closeCommandPalette();
    } else if (e.code === "Space" && state.currentView === "view-query-lab" && document.activeElement.tagName !== "INPUT") {
      e.preventDefault();
      triggerQuerySearch();
    }
  });

  if (searchModal) {
    searchModal.addEventListener("click", (e) => {
      if (e.target === searchModal) closeCommandPalette();
    });
  }

  cmdItems.forEach(item => {
    item.addEventListener("click", () => {
      const action = item.getAttribute("data-action");
      if (action === "overview") switchView("view-overview");
      else if (action === "graph") switchView("view-graph-explorer");
      else if (action === "query") switchView("view-query-lab");
      else if (action === "bench") switchView("view-benchmarks");
      closeCommandPalette();
    });
  });

  // ==========================================================================
  // GENERATE SYNTHETIC HIGH-DIMENSIONAL RESEARCH DATA
  // ==========================================================================
  function generateSyntheticGraph() {
    const nodes = [];
    const edges = [];
    const N = 120;

    // Fixed key node #48,219
    nodes.push({
      id: 48219,
      x: 0.65,
      y: 0.45,
      lid: 26.4,
      density: 0.041,
      m: 22,
      layer: 0,
      isTarget: true,
      tag: "MANIFOLD CREST"
    });

    // Cluster 1: Dense Core (Low LID ~8-14, High density ~0.15, M ~10-12)
    for (let i = 0; i < 45; i++) {
      const r = Math.sqrt(Math.random()) * 0.18;
      const th = Math.random() * Math.PI * 2;
      nodes.push({
        id: 10000 + i,
        x: 0.28 + r * Math.cos(th),
        y: 0.68 + r * Math.sin(th),
        lid: 8.5 + Math.random() * 4.5,
        density: 0.12 + Math.random() * 0.08,
        m: 10 + Math.floor(Math.random() * 4),
        layer: Math.random() < 0.15 ? 1 : 0
      });
    }

    // Cluster 2: Manifold Crest & Boundary Zone (High LID ~20-30, Low density, M ~20-24)
    for (let i = 0; i < 40; i++) {
      const r = Math.sqrt(Math.random()) * 0.22;
      const th = Math.random() * Math.PI * 2;
      nodes.push({
        id: 48000 + i,
        x: 0.68 + r * Math.cos(th),
        y: 0.40 + r * Math.sin(th),
        lid: 22.0 + Math.random() * 7.5,
        density: 0.035 + Math.random() * 0.03,
        m: 20 + Math.floor(Math.random() * 5),
        layer: Math.random() < 0.25 ? 1 : 0
      });
    }

    // Outlier Bridge (Highway nodes connecting clusters)
    for (let i = 0; i < 20; i++) {
      const t = i / 20;
      nodes.push({
        id: 20000 + i,
        x: 0.32 + t * 0.32 + (Math.random() - 0.5) * 0.08,
        y: 0.65 - t * 0.22 + (Math.random() - 0.5) * 0.08,
        lid: 16.0 + Math.random() * 5.0,
        density: 0.07 + Math.random() * 0.04,
        m: 16 + Math.floor(Math.random() * 4),
        layer: Math.random() < 0.4 ? 2 : 1
      });
    }

    // Generate K=4 sparse edges
    for (let i = 0; i < nodes.length; i++) {
      const src = nodes[i];
      // Find nearest 3-4 nodes
      const dists = [];
      for (let j = 0; j < nodes.length; j++) {
        if (i === j) continue;
        const dst = nodes[j];
        const d = Math.hypot(src.x - dst.x, src.y - dst.y);
        dists.push({ j, d });
      }
      dists.sort((a, b) => a.d - b.d);
      const k = Math.min(3, dists.length);
      for (let e = 0; e < k; e++) {
        edges.push([i, dists[e].j]);
      }
    }

    state.overview.nodes = nodes;
    state.overview.edges = edges;
  }

  generateSyntheticGraph();

  // ==========================================================================
  // PAGE 1: OVERVIEW 2D CANVAS (Adaptive Graph Profile)
  // ==========================================================================
  const overviewCanvas = document.getElementById("overview-graph-canvas");
  const overviewCtx = overviewCanvas ? overviewCanvas.getContext("2d") : null;

  function resizeCanvasToDisplaySize(canvas) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.round(rect.width * dpr);
    const displayHeight = Math.round(rect.height * dpr);
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
  }

  function getNodeColor(node) {
    if (node.isTarget) return "#00e5ff"; // Bright cyan for #48,219
    if (state.colorMode === "lid") {
      // Low LID: sapphire/emerald, High LID: coral/rose
      const norm = Math.min(1, Math.max(0, (node.lid - 8) / 22));
      if (norm < 0.5) return "#38bdf8";
      return "#f43f5e";
    } else if (state.colorMode === "density") {
      // Density: dense = amber, sparse = slate
      return node.density > 0.08 ? "#fbbf24" : "#64748b";
    } else {
      // Adaptive M: M=8..12 blue, M=16 cyan, M=20..24 emerald
      if (node.m <= 12) return "#38bdf8";
      if (node.m <= 16) return "#06b6d4";
      return "#10b981";
    }
  }

  function renderOverviewCanvas() {
    if (!overviewCanvas || !overviewCtx) return;
    resizeCanvasToDisplaySize(overviewCanvas);

    const w = overviewCanvas.width;
    const h = overviewCanvas.height;
    overviewCtx.clearRect(0, 0, w, h);

    // Save transform
    overviewCtx.save();
    overviewCtx.translate(state.overview.panX, state.overview.panY);
    overviewCtx.scale(state.overview.zoom, state.overview.zoom);

    // Draw Subtle Grid Lines
    overviewCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    overviewCtx.lineWidth = 1;
    const gridSize = 40 * (window.devicePixelRatio || 1);
    for (let x = 0; x < w; x += gridSize) {
      overviewCtx.beginPath();
      overviewCtx.moveTo(x, 0);
      overviewCtx.lineTo(x, h);
      overviewCtx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
      overviewCtx.beginPath();
      overviewCtx.moveTo(0, y);
      overviewCtx.lineTo(w, y);
      overviewCtx.stroke();
    }

    const pad = 40;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;

    const nodes = state.overview.nodes;
    const edges = state.overview.edges;

    // Draw Edges
    overviewCtx.strokeStyle = "rgba(6, 182, 212, 0.15)";
    overviewCtx.lineWidth = 1;
    for (let i = 0; i < edges.length; i++) {
      const [u, v] = edges[i];
      const p1 = nodes[u];
      const p2 = nodes[v];
      if (!p1 || !p2) continue;
      const x1 = pad + p1.x * plotW;
      const y1 = pad + p1.y * plotH;
      const x2 = pad + p2.x * plotW;
      const y2 = pad + p2.y * plotH;

      overviewCtx.beginPath();
      overviewCtx.moveTo(x1, y1);
      overviewCtx.lineTo(x2, y2);
      overviewCtx.stroke();
    }

    // Draw Nodes
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const cx = pad + n.x * plotW;
      const cy = pad + n.y * plotH;
      const radius = (n.m / 24) * 5 + 2.5;

      const isSelected = n.id === state.overview.selectedNode;

      if (isSelected || n.isTarget) {
        // Glowing halo for selected/target node
        overviewCtx.beginPath();
        overviewCtx.arc(cx, cy, radius + 12, 0, Math.PI * 2);
        overviewCtx.fillStyle = "rgba(0, 229, 255, 0.15)";
        overviewCtx.fill();

        overviewCtx.beginPath();
        overviewCtx.arc(cx, cy, radius + 6, 0, Math.PI * 2);
        overviewCtx.strokeStyle = "rgba(0, 229, 255, 0.7)";
        overviewCtx.lineWidth = 1.5;
        overviewCtx.setLineDash([3, 3]);
        overviewCtx.stroke();
        overviewCtx.setLineDash([]);
      }

      overviewCtx.beginPath();
      overviewCtx.arc(cx, cy, radius, 0, Math.PI * 2);
      overviewCtx.fillStyle = getNodeColor(n);
      overviewCtx.fill();
    }

    overviewCtx.restore();
  }

  // Interactivity on Overview Canvas (Pan, Zoom, Node Click)
  if (overviewCanvas) {
    overviewCanvas.addEventListener("mousedown", (e) => {
      state.overview.isDragging = true;
      state.overview.dragStartX = e.clientX - state.overview.panX;
      state.overview.dragStartY = e.clientY - state.overview.panY;
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.overview.isDragging) return;
      state.overview.panX = e.clientX - state.overview.dragStartX;
      state.overview.panY = e.clientY - state.overview.dragStartY;
      renderOverviewCanvas();
    });

    window.addEventListener("mouseup", () => {
      state.overview.isDragging = false;
    });

    overviewCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      state.overview.zoom = Math.max(0.6, Math.min(3.0, state.overview.zoom * zoomFactor));
      renderOverviewCanvas();
    });

    overviewCanvas.addEventListener("click", (e) => {
      const rect = overviewCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const mouseX = (e.clientX - rect.left) * dpr;
      const mouseY = (e.clientY - rect.top) * dpr;

      const pad = 40;
      const plotW = overviewCanvas.width - pad * 2;
      const plotH = overviewCanvas.height - pad * 2;

      // Find closest node
      let closest = null;
      let minD = 25 * dpr;

      state.overview.nodes.forEach(n => {
        const cx = (pad + n.x * plotW) * state.overview.zoom + state.overview.panX;
        const cy = (pad + n.y * plotH) * state.overview.zoom + state.overview.panY;
        const d = Math.hypot(mouseX - cx, mouseY - cy);
        if (d < minD) {
          minD = d;
          closest = n;
        }
      });

      if (closest) {
        state.overview.selectedNode = closest.id;
        playHapticBeep(920, 0.04);
        updateOverviewCallout(closest);
        renderOverviewCanvas();
      }
    });
  }

  function updateOverviewCallout(node) {
    const idEl = document.getElementById("callout-node-id");
    const tagEl = document.getElementById("callout-tag");
    const lidEl = document.getElementById("callout-lid");
    const densEl = document.getElementById("callout-density");
    if (idEl) idEl.textContent = `Node #${node.id.toLocaleString()}`;
    if (tagEl) tagEl.textContent = node.tag || (node.lid > 20 ? "MANIFOLD CREST" : "CLUSTERED CORE");
    if (lidEl) lidEl.textContent = node.lid.toFixed(1);
    if (densEl) densEl.textContent = node.density.toFixed(3);
  }

  // Projection / Color Mode Segmented Buttons
  const segProj = document.querySelectorAll("#seg-projection .pill-btn");
  segProj.forEach(btn => {
    btn.addEventListener("click", () => {
      segProj.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.projectionMode = btn.getAttribute("data-proj");
      playHapticBeep(700, 0.03);
      renderOverviewCanvas();
    });
  });

  const segColor = document.querySelectorAll("#seg-color-mode .pill-btn");
  segColor.forEach(btn => {
    btn.addEventListener("click", () => {
      segColor.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.colorMode = btn.getAttribute("data-mode");
      playHapticBeep(740, 0.03);
      renderOverviewCanvas();
    });
  });

  const btnResetOverview = document.getElementById("btn-reset-overview-view");
  if (btnResetOverview) {
    btnResetOverview.addEventListener("click", () => {
      state.overview.panX = 0;
      state.overview.panY = 0;
      state.overview.zoom = 1;
      renderOverviewCanvas();
    });
  }

  // Draw Diagnostic 2: Density Curve
  function drawDensityCurve() {
    const cvs = document.getElementById("density-curve-canvas");
    if (!cvs) return;
    resizeCanvasToDisplaySize(cvs);
    const ctx = cvs.getContext("2d");
    const w = cvs.width;
    const h = cvs.height;
    ctx.clearRect(0, 0, w, h);

    ctx.beginPath();
    ctx.moveTo(0, h - 5);
    // Skewed bell curve
    ctx.bezierCurveTo(w * 0.2, h * 0.1, w * 0.45, h * 0.05, w * 0.7, h * 0.7);
    ctx.bezierCurveTo(w * 0.85, h * 0.9, w * 0.95, h - 5, w, h - 5);

    // Gradient stroke
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "#06b6d4");
    grad.addColorStop(0.5, "#00e5ff");
    grad.addColorStop(1, "#38bdf8");

    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(6, 182, 212, 0.12)";
    ctx.fill();
  }

  // ==========================================================================
  // PAGE 2: GRAPH EXPLORER (Geodesic Polar Routing Canvas)
  // ==========================================================================
  const geodesicCanvas = document.getElementById("geodesic-canvas");
  const geodesicCtx = geodesicCanvas ? geodesicCanvas.getContext("2d") : null;

  function renderGeodesicCanvas() {
    if (!geodesicCanvas || !geodesicCtx) return;
    resizeCanvasToDisplaySize(geodesicCanvas);

    const w = geodesicCanvas.width;
    const h = geodesicCanvas.height;
    geodesicCtx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(w, h) * 0.44;

    // Concentric Geodesic Distance Rings (Matching Screenshot 2: r=0.90, 0.65, 0.40, 0.20)
    const rings = [
      { r: 0.90, label: "r = 0.90 (Highway Frontier)" },
      { r: 0.65, label: "r = 0.65 (L2)" },
      { r: 0.40, label: "r = 0.40 (L1)" },
      { r: 0.20, label: "r = 0.20 (L0)" }
    ];

    rings.forEach(ring => {
      const radius = ring.r * maxR;
      geodesicCtx.beginPath();
      geodesicCtx.arc(cx, cy, radius, 0, Math.PI * 2);
      geodesicCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      geodesicCtx.lineWidth = 1;
      geodesicCtx.stroke();

      // Ring label
      geodesicCtx.font = "10px 'JetBrains Mono', monospace";
      geodesicCtx.fillStyle = "rgba(255, 255, 255, 0.25)";
      geodesicCtx.fillText(ring.label, cx + 8, cy - radius + 12);
    });

    // Crosshairs
    geodesicCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    geodesicCtx.beginPath();
    geodesicCtx.moveTo(cx, cy - maxR);
    geodesicCtx.lineTo(cx, cy + maxR);
    geodesicCtx.moveTo(cx - maxR, cy);
    geodesicCtx.lineTo(cx + maxR, cy);
    geodesicCtx.stroke();

    // Hops coordinates in polar space relative to center
    const hops = [
      { id: 10402, r: 0.85, angle: Math.PI * 1.25, label: "ENTRY #10,402\n(Layer 4, Top)" },
      { id: 12890, r: 0.58, angle: Math.PI * 1.30, label: "HOP 1" },
      { id: 34011, r: 0.32, angle: Math.PI * 1.38, label: "HOP 2" },
      { id: 48219, r: 0.05, angle: Math.PI * 1.45, label: "TARGET #48,219" }
    ];

    // Background Ghost Neighbors
    const ghostNodes = [
      { id: 53819, r: 0.28, angle: 0.3 },
      { id: 46201, r: 0.35, angle: 1.1 },
      { id: 71092, r: 0.75, angle: 0.15 },
      { id: 26014, r: 0.68, angle: 3.4 },
      { id: 88290, r: 0.42, angle: 4.8 }
    ];

    // Draw Ghost Baseline Edges (Dashed)
    geodesicCtx.strokeStyle = "rgba(244, 63, 94, 0.25)";
    geodesicCtx.setLineDash([4, 4]);
    ghostNodes.forEach(gn => {
      const gx = cx + gn.r * maxR * Math.cos(gn.angle);
      const gy = cy + gn.r * maxR * Math.sin(gn.angle);
      geodesicCtx.beginPath();
      geodesicCtx.moveTo(cx, cy);
      geodesicCtx.lineTo(gx, gy);
      geodesicCtx.stroke();

      geodesicCtx.beginPath();
      geodesicCtx.arc(gx, gy, 4, 0, Math.PI * 2);
      geodesicCtx.fillStyle = "rgba(244, 63, 94, 0.6)";
      geodesicCtx.fill();
    });
    geodesicCtx.setLineDash([]);

    // Draw Active Cyan Traversal Path
    geodesicCtx.strokeStyle = "#00e5ff";
    geodesicCtx.lineWidth = 2.5;
    geodesicCtx.beginPath();

    const hopCoords = hops.map(h => {
      return {
        x: cx + h.r * maxR * Math.cos(h.angle),
        y: cy + h.r * maxR * Math.sin(h.angle),
        ...h
      };
    });

    for (let i = 0; i < hopCoords.length; i++) {
      if (i === 0) geodesicCtx.moveTo(hopCoords[i].x, hopCoords[i].y);
      else geodesicCtx.lineTo(hopCoords[i].x, hopCoords[i].y);
    }
    geodesicCtx.stroke();

    // Draw Hop Nodes & Rings
    hopCoords.forEach((node, idx) => {
      // Outer ripple
      geodesicCtx.beginPath();
      geodesicCtx.arc(node.x, node.y, 14, 0, Math.PI * 2);
      geodesicCtx.strokeStyle = "rgba(0, 229, 255, 0.4)";
      geodesicCtx.lineWidth = 1;
      geodesicCtx.stroke();

      // Core Node
      geodesicCtx.beginPath();
      geodesicCtx.arc(node.x, node.y, 6, 0, Math.PI * 2);
      geodesicCtx.fillStyle = idx === hopCoords.length - 1 ? "#00e5ff" : "#38bdf8";
      geodesicCtx.fill();

      // Label text
      geodesicCtx.font = "11px 'JetBrains Mono', monospace";
      geodesicCtx.fillStyle = "#fff";
      geodesicCtx.fillText(node.label, node.x + 12, node.y - 6);
    });
  }

  // Layer Filter Buttons
  const layerBtns = document.querySelectorAll(".layer-btn");
  layerBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      layerBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeLayer = parseInt(btn.getAttribute("data-layer") || "0", 10);
      playHapticBeep(800, 0.03);
      renderGeodesicCanvas();
    });
  });

  // Jump Target Node Button
  const btnJumpTarget = document.getElementById("btn-jump-target");
  const inputTargetNode = document.getElementById("input-target-node");
  if (btnJumpTarget && inputTargetNode) {
    btnJumpTarget.addEventListener("click", () => {
      const val = inputTargetNode.value.replace(/[^0-9]/g, "");
      if (val) {
        state.targetNodeId = parseInt(val, 10);
        const titleEl = document.getElementById("insp-target-title");
        if (titleEl) titleEl.textContent = `Node #${state.targetNodeId.toLocaleString()}`;
        playHapticBeep(980, 0.05);
        renderGeodesicCanvas();
      }
    });
  }

  // ==========================================================================
  // PAGE 3: QUERY LAB (Real-Time Retrieval & Frontier Analysis)
  // ==========================================================================
  const trajectoryCanvas = document.getElementById("trajectory-convergence-canvas");
  const trajectoryCtx = trajectoryCanvas ? trajectoryCanvas.getContext("2d") : null;

  function renderTrajectoryChart() {
    if (!trajectoryCanvas || !trajectoryCtx) return;
    resizeCanvasToDisplaySize(trajectoryCanvas);

    const w = trajectoryCanvas.width;
    const h = trajectoryCanvas.height;
    trajectoryCtx.clearRect(0, 0, w, h);

    const padLeft = 60;
    const padRight = 30;
    const padTop = 30;
    const padBottom = 40;
    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    // Background horizontal grid lines
    const yTicks = [0.04, 0.30, 0.60, 0.95];
    yTicks.forEach(tick => {
      const normY = 1 - (tick - 0.04) / (0.95 - 0.04);
      const y = padTop + normY * chartH;

      trajectoryCtx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      trajectoryCtx.lineWidth = 1;
      trajectoryCtx.beginPath();
      trajectoryCtx.moveTo(padLeft, y);
      trajectoryCtx.lineTo(w - padRight, y);
      trajectoryCtx.stroke();

      trajectoryCtx.font = "10px 'JetBrains Mono', monospace";
      trajectoryCtx.fillStyle = "rgba(255, 255, 255, 0.35)";
      trajectoryCtx.fillText(`d=${tick.toFixed(2)}`, 10, y + 3);
    });

    // X-axis Hop Ticks (Hop 0, 4, 8, 12, 16, 20, 24)
    const hopsList = [0, 4, 8, 12, 16, 20, 24];
    hopsList.forEach(hop => {
      const x = padLeft + (hop / 24) * chartW;
      trajectoryCtx.font = "10px 'JetBrains Mono', monospace";
      trajectoryCtx.fillStyle = hop === 16 ? "#00e5ff" : "rgba(255, 255, 255, 0.35)";
      const label = hop === 16 ? "Hop 16 (Halt)" : `Hop ${hop}`;
      trajectoryCtx.fillText(label, x - 18, h - 14);
    });

    const data = state.query.trajectory;

    // 1. Draw Baseline HNSW Curve (Dashed Muted Line)
    trajectoryCtx.strokeStyle = "rgba(148, 163, 184, 0.5)";
    trajectoryCtx.lineWidth = 1.5;
    trajectoryCtx.setLineDash([4, 4]);
    trajectoryCtx.beginPath();
    data.forEach((pt, idx) => {
      const x = padLeft + (pt.hop / 24) * chartW;
      const normY = 1 - (pt.baselineDist - 0.04) / (0.95 - 0.04);
      const y = padTop + normY * chartH;
      if (idx === 0) trajectoryCtx.moveTo(x, y);
      else trajectoryCtx.lineTo(x, y);
    });
    trajectoryCtx.stroke();
    trajectoryCtx.setLineDash([]);

    // 2. Draw AdaptiveVec Curve (Cyan Glow Line)
    trajectoryCtx.strokeStyle = "#00e5ff";
    trajectoryCtx.lineWidth = 2.5;
    trajectoryCtx.beginPath();
    const adaptiveData = data.filter(pt => pt.hop <= 16);
    adaptiveData.forEach((pt, idx) => {
      const x = padLeft + (pt.hop / 24) * chartW;
      const normY = 1 - (pt.adaptiveDist - 0.04) / (0.95 - 0.04);
      const y = padTop + normY * chartH;
      if (idx === 0) trajectoryCtx.moveTo(x, y);
      else trajectoryCtx.lineTo(x, y);
    });
    trajectoryCtx.stroke();

    // Draw Hop Points along AdaptiveVec Curve
    adaptiveData.forEach((pt) => {
      const x = padLeft + (pt.hop / 24) * chartW;
      const normY = 1 - (pt.adaptiveDist - 0.04) / (0.95 - 0.04);
      const y = padTop + normY * chartH;

      trajectoryCtx.beginPath();
      trajectoryCtx.arc(x, y, pt.hop === 16 ? 5.5 : 3.5, 0, Math.PI * 2);
      trajectoryCtx.fillStyle = pt.hop === 16 ? "#00e5ff" : "#38bdf8";
      trajectoryCtx.fill();
    });

    // 3. Early Exit Trigger Banner Overlay at Hop 16
    const exitX = padLeft + (16 / 24) * chartW;
    const exitNormY = 1 - (0.0418 - 0.04) / (0.95 - 0.04);
    const exitY = padTop + exitNormY * chartH;

    // Callout box
    trajectoryCtx.fillStyle = "rgba(14, 19, 31, 0.95)";
    trajectoryCtx.strokeStyle = "rgba(6, 182, 212, 0.7)";
    trajectoryCtx.lineWidth = 1;
    trajectoryCtx.strokeRect(exitX - 60, exitY - 70, 150, 48);
    trajectoryCtx.fillRect(exitX - 60, exitY - 70, 150, 48);

    trajectoryCtx.font = "bold 10px 'JetBrains Mono', monospace";
    trajectoryCtx.fillStyle = "#00e5ff";
    trajectoryCtx.fillText("EARLY EXIT TRIGGERED", exitX - 52, exitY - 50);
    trajectoryCtx.font = "9px 'JetBrains Mono', monospace";
    trajectoryCtx.fillStyle = "#94a3b8";
    trajectoryCtx.fillText("(HOP 16: Δ < 1e-4)", exitX - 52, exitY - 34);

    // Stagnation baseline exploration zone
    trajectoryCtx.font = "9.5px 'JetBrains Mono', monospace";
    trajectoryCtx.fillStyle = "rgba(148, 163, 184, 0.4)";
    trajectoryCtx.fillText("Stagnation Zone: ε < 0.0001 (Unnecessary Baseline Exploration)", exitX + 15, exitY + 18);
  }

  // Live Query Search Execution
  const btnRunQuerySearch = document.getElementById("btn-run-query-search");
  function triggerQuerySearch() {
    if (state.isSearching) return;
    state.isSearching = true;
    playHapticBeep(950, 0.06);

    const latencyEl = document.getElementById("ql-last-latency");
    if (btnRunQuerySearch) {
      btnRunQuerySearch.innerHTML = `<span class="dot-em"></span> SEARCHING...`;
    }

    // Simulate search latency progression
    setTimeout(() => {
      const simulatedLatency = (1.05 + Math.random() * 0.25).toFixed(2);
      if (latencyEl) latencyEl.textContent = `${simulatedLatency} ms`;
      if (btnRunQuerySearch) {
        btnRunQuerySearch.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width:14px;height:14px;"><polygon points="5 3 19 12 5 21 5 3"/></svg> RUN SEARCH <span class="kbd-chip-dark">Space</span>`;
      }
      state.isSearching = false;
      playHapticBeep(1100, 0.04);
      renderTrajectoryChart();
    }, 280);
  }

  if (btnRunQuerySearch) {
    btnRunQuerySearch.addEventListener("click", triggerQuerySearch);
  }

  // K and efSearch Sliders
  const sliderK = document.getElementById("ql-k-slider");
  const valK = document.getElementById("ql-k-val");
  if (sliderK && valK) {
    sliderK.addEventListener("input", (e) => {
      valK.textContent = `${e.target.value} pts`;
      state.kNeighbors = parseInt(e.target.value, 10);
    });
  }

  const sliderEf = document.getElementById("ql-ef-slider");
  const valEf = document.getElementById("ql-ef-val");
  if (sliderEf && valEf) {
    sliderEf.addEventListener("input", (e) => {
      valEf.textContent = e.target.value;
      state.efSearch = parseInt(e.target.value, 10);
    });
  }

  // ==========================================================================
  // PAGE 4: BENCHMARKS (Pareto Frontier & Memory Scaling Canvas)
  // ==========================================================================
  const paretoCanvas = document.getElementById("pareto-chart-canvas");
  const paretoCtx = paretoCanvas ? paretoCanvas.getContext("2d") : null;

  function renderBenchmarkCharts() {
    if (!paretoCanvas || !paretoCtx) return;
    resizeCanvasToDisplaySize(paretoCanvas);

    const w = paretoCanvas.width;
    const h = paretoCanvas.height;
    paretoCtx.clearRect(0, 0, w, h);

    const padL = 50;
    const padR = 30;
    const padT = 25;
    const padB = 35;
    const cW = w - padL - padR;
    const cH = h - padT - padB;

    // Y Axis: QPS (2k to 10k)
    const yTicks = [2000, 4000, 6000, 8000, 10000];
    yTicks.forEach(tick => {
      const normY = 1 - (tick - 2000) / 8000;
      const y = padT + normY * cH;
      paretoCtx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      paretoCtx.beginPath();
      paretoCtx.moveTo(padL, y);
      paretoCtx.lineTo(w - padR, y);
      paretoCtx.stroke();

      paretoCtx.font = "10px 'JetBrains Mono', monospace";
      paretoCtx.fillStyle = "rgba(255, 255, 255, 0.35)";
      paretoCtx.fillText(`${tick / 1000}k`, 15, y + 3);
    });

    // X Axis: Recall@10 (0.90 to 1.00)
    const xTicks = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00];
    xTicks.forEach(tick => {
      const normX = (tick - 0.90) / 0.10;
      const x = padL + normX * cW;
      paretoCtx.font = "10px 'JetBrains Mono', monospace";
      paretoCtx.fillStyle = "rgba(255, 255, 255, 0.35)";
      paretoCtx.fillText(tick.toFixed(2), x - 10, h - 10);
    });

    // 1. Baseline HNSW Fixed M=32 (Dotted Gray)
    paretoCtx.strokeStyle = "rgba(148, 163, 184, 0.55)";
    paretoCtx.setLineDash([4, 4]);
    paretoCtx.lineWidth = 1.8;
    paretoCtx.beginPath();
    const basePts = [
      { r: 0.91, q: 7200 },
      { r: 0.94, q: 6800 },
      { r: 0.97, q: 5800 },
      { r: 0.989, q: 4120 }
    ];
    basePts.forEach((pt, i) => {
      const x = padL + ((pt.r - 0.90) / 0.10) * cW;
      const y = padT + (1 - (pt.q - 2000) / 8000) * cH;
      if (i === 0) paretoCtx.moveTo(x, y);
      else paretoCtx.lineTo(x, y);
    });
    paretoCtx.stroke();
    paretoCtx.setLineDash([]);

    // 2. AdaptiveVec Pareto Frontier Curve (Bright Cyan)
    paretoCtx.strokeStyle = "#00e5ff";
    paretoCtx.lineWidth = 2.5;
    paretoCtx.beginPath();
    const adaptPts = [
      { r: 0.915, q: 9400 },
      { r: 0.945, q: 8900 },
      { r: 0.972, q: 8400 },
      { r: 0.987, q: 7840 }
    ];
    adaptPts.forEach((pt, i) => {
      const x = padL + ((pt.r - 0.90) / 0.10) * cW;
      const y = padT + (1 - (pt.q - 2000) / 8000) * cH;
      if (i === 0) paretoCtx.moveTo(x, y);
      else paretoCtx.lineTo(x, y);
    });
    paretoCtx.stroke();

    // Highlight key Pareto Point (7,840 QPS @ 98.7% Recall)
    const keyPt = adaptPts[3];
    const kx = padL + ((keyPt.r - 0.90) / 0.10) * cW;
    const ky = padT + (1 - (keyPt.q - 2000) / 8000) * cH;

    paretoCtx.beginPath();
    paretoCtx.arc(kx, ky, 6, 0, Math.PI * 2);
    paretoCtx.fillStyle = "#00e5ff";
    paretoCtx.fill();

    // Callout text
    paretoCtx.fillStyle = "rgba(14, 19, 31, 0.9)";
    paretoCtx.strokeStyle = "rgba(6, 182, 212, 0.5)";
    paretoCtx.strokeRect(kx - 130, ky - 30, 120, 24);
    paretoCtx.fillRect(kx - 130, ky - 30, 120, 24);
    paretoCtx.font = "bold 9.5px 'JetBrains Mono', monospace";
    paretoCtx.fillStyle = "#00e5ff";
    paretoCtx.fillText("AdaptiveVec: 7,840 QPS", kx - 124, ky - 14);

    // Render Memory Scaling Canvas
    renderMemoryScalingCanvas();
  }

  function renderMemoryScalingCanvas() {
    const memCanvas = document.getElementById("memory-scaling-canvas");
    if (!memCanvas) return;
    resizeCanvasToDisplaySize(memCanvas);
    const ctx = memCanvas.getContext("2d");
    const w = memCanvas.width;
    const h = memCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 50;
    const padR = 30;
    const padT = 30;
    const padB = 35;
    const cW = w - padL - padR;
    const cH = h - padT - padB;

    // Horizontal Memory Comparison line
    const y = padT + cH * 0.45;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL + cW * 0.25, y);
    ctx.lineTo(padL + cW * 0.85, y);
    ctx.stroke();

    // AdaptiveVec point (214.6 MB)
    const ax = padL + cW * 0.25;
    ctx.beginPath();
    ctx.arc(ax, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#00e5ff";
    ctx.fill();

    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#00e5ff";
    ctx.fillText("AdaptiveVec: 214.6 MB", ax - 45, y - 14);

    // Baseline point (368.0 MB)
    const bx = padL + cW * 0.85;
    ctx.beginPath();
    ctx.arc(bx, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#94a3b8";
    ctx.fill();

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("Baseline: 368.0 MB", bx - 40, y - 14);

    // Savings bracket & text
    ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
    ctx.strokeRect(ax, y + 10, bx - ax, 28);
    ctx.fillStyle = "rgba(6, 182, 212, 0.08)";
    ctx.fillRect(ax, y + 10, bx - ax, 28);

    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#10b981";
    ctx.fillText("-41.7% MEMORY SAVED (ΔRecall: -0.19%)", ax + 25, y + 28);
  }

  // Initial draw
  requestAnimationFrame(() => {
    renderOverviewCanvas();
    drawDensityCurve();
  });

  // Window resize handler
  window.addEventListener("resize", () => {
    if (state.currentView === "view-overview") {
      renderOverviewCanvas();
      drawDensityCurve();
    } else if (state.currentView === "view-graph-explorer") {
      renderGeodesicCanvas();
    } else if (state.currentView === "view-query-lab") {
      renderTrajectoryChart();
    } else if (state.currentView === "view-benchmarks") {
      renderBenchmarkCharts();
    }
  });
});
