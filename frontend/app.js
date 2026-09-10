/**
 * AdaptiveVec Interactive Frontend Studio & Visualizer Engine
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    activeTab: "tab-overview",
    datasetMeta: null,
    graphData: null,
    colorMode: "lid",
    simTrace: null,
    simStepIndex: 0,
    simPlaying: false,
    simTimer: null,
    benchmarkResults: [],
    chartMemory: null,
    chartQps: null,
    zoom: { x: 0, y: 0, scale: 1, isDragging: false, startX: 0, startY: 0 }
  };

  // DOM Elements
  const tabButtons = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const graphCanvas = document.getElementById("graph-canvas");
  const graphCtx = graphCanvas ? graphCanvas.getContext("2d") : null;
  const simCanvas = document.getElementById("sim-canvas");
  const simCtx = simCanvas ? simCanvas.getContext("2d") : null;
  const tooltip = document.getElementById("canvas-tooltip");

  // Initialize
  initTabs();
  initCanvasEvents();
  fetchStatus();
  fetchGraphProjection();

  // Tab Switching
  function initTabs() {
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        tabButtons.forEach(b => b.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const activePane = document.getElementById(targetTab);
        if (activePane) activePane.classList.add("active");
        state.activeTab = targetTab;

        if (targetTab === "tab-visualizer") {
          renderGraph();
        } else if (targetTab === "tab-simulator") {
          renderSimulationStep();
        }
      });
    });

    // Quick links
    const btnSwitchViz = document.getElementById("btn-switch-viz");
    if (btnSwitchViz) {
      btnSwitchViz.addEventListener("click", () => {
        const vizTabBtn = document.getElementById("btn-tab-visualizer");
        if (vizTabBtn) vizTabBtn.click();
      });
    }
  }

  // Fetch API Status & Stats
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      state.datasetMeta = data.dataset;

      document.getElementById("val-nodes").textContent = (data.n_samples || 0).toLocaleString();
      document.getElementById("val-dim").textContent = `${data.dim || 32}-D`;

      if (data.adaptive_stats && data.stock_stats) {
        const stockEdges = data.stock_stats.total_edges || 1;
        const adaptEdges = data.adaptive_stats.total_edges || 0;
        const savings = Math.max(0, ((stockEdges - adaptEdges) / stockEdges) * 100).toFixed(1);
        document.getElementById("val-savings").textContent = `-${savings}%`;

        const stockTime = data.stock_stats.build_time_s || 1;
        const adaptTime = data.adaptive_stats.build_time_s || 1;
        const speedup = Math.max(0, ((stockTime - adaptTime) / stockTime) * 100).toFixed(1);
        document.getElementById("val-speedup").textContent = `+${speedup}%`;
      }
    } catch (err) {
      console.error("Status fetch error:", err);
    }
  }

  // Fetch 2D Graph Projection Data
  async function fetchGraphProjection() {
    try {
      const res = await fetch("/api/graph/projection");
      const data = await res.json();
      state.graphData = data;
      updateLegendBounds();
      renderGraph();
    } catch (err) {
      console.error("Graph projection fetch error:", err);
    }
  }

  // Update Legend Min / Max Labels based on color mode
  function updateLegendBounds() {
    if (!state.graphData || !state.graphData.nodes) return;
    const nodes = state.graphData.nodes;
    const mode = state.colorMode;

    let vals = nodes.map(n => n[mode] || 0);
    let minVal = Math.min(...vals);
    let maxVal = Math.max(...vals);

    document.getElementById("legend-min").textContent = `Min: ${minVal.toFixed(1)}`;
    document.getElementById("legend-max").textContent = `Max: ${maxVal.toFixed(1)}`;
  }

  // Color Mapping Helper
  function getNodeColor(node, mode) {
    if (!state.graphData || !state.graphData.nodes) return "#6366f1";
    
    if (mode === "level") {
      if (node.level >= 2) return "#fbbf24"; // Gold for high express layers
      if (node.level === 1) return "#38bdf8"; // Sky blue
      return "#6366f1"; // Indigo for ground layer
    }

    const val = node[mode] !== undefined ? node[mode] : 0;
    
    // Normalize between min and max
    let min = 0, max = 10;
    if (mode === "lid") { min = 1; max = 16; }
    else if (mode === "density") { min = 0.2; max = 5.0; }
    else if (mode === "degree") { min = 4; max = 32; }
    else if (mode === "score") { min = -2.0; max = 2.0; }

    const norm = Math.max(0, Math.min(1, (val - min) / (max - min || 1)));

    // Interpolate Cyan (#06b6d4) -> Indigo (#6366f1) -> Rose (#f43f5e)
    if (norm < 0.5) {
      const t = norm * 2;
      const r = Math.round(6 + t * (99 - 6));
      const g = Math.round(182 + t * (102 - 182));
      const b = Math.round(212 + t * (241 - 212));
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = (norm - 0.5) * 2;
      const r = Math.round(99 + t * (244 - 99));
      const g = Math.round(102 + t * (63 - 102));
      const b = Math.round(241 + t * (94 - 241));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  // Render 2D Graph onto Canvas
  function renderGraph() {
    if (!graphCtx || !state.graphData || !state.graphData.nodes) return;

    const width = graphCanvas.width;
    const height = graphCanvas.height;
    graphCtx.clearRect(0, 0, width, height);

    graphCtx.save();
    graphCtx.translate(width / 2 + state.zoom.x, height / 2 + state.zoom.y);
    graphCtx.scale(state.zoom.scale, state.zoom.scale);

    const scaleFactor = Math.min(width, height) * 0.42;

    const showL0Edges = document.getElementById("toggle-layer0-edges")?.checked ?? true;
    const showL1Edges = document.getElementById("toggle-layer1-edges")?.checked ?? true;

    // 1. Draw Layer 0 Proximity Edges
    if (showL0Edges && state.graphData.edges_l0) {
      graphCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      graphCtx.lineWidth = 0.75;
      graphCtx.beginPath();
      for (const e of state.graphData.edges_l0) {
        const u = state.graphData.nodes[e.source];
        const v = state.graphData.nodes[e.target];
        if (u && v) {
          graphCtx.moveTo(u.x * scaleFactor, u.y * scaleFactor);
          graphCtx.lineTo(v.x * scaleFactor, v.y * scaleFactor);
        }
      }
      graphCtx.stroke();
    }

    // 2. Draw Layer 1 Express Links
    if (showL1Edges && state.graphData.edges_l1) {
      graphCtx.strokeStyle = "rgba(6, 182, 212, 0.35)";
      graphCtx.lineWidth = 1.6;
      graphCtx.beginPath();
      for (const e of state.graphData.edges_l1) {
        const u = state.graphData.nodes[e.source];
        const v = state.graphData.nodes[e.target];
        if (u && v) {
          graphCtx.moveTo(u.x * scaleFactor, u.y * scaleFactor);
          graphCtx.lineTo(v.x * scaleFactor, v.y * scaleFactor);
        }
      }
      graphCtx.stroke();
    }

    // 3. Draw Nodes
    for (const node of state.graphData.nodes) {
      const nx = node.x * scaleFactor;
      const ny = node.y * scaleFactor;
      const radius = node.level >= 2 ? 4.5 : (node.level === 1 ? 3.5 : 2.5);

      graphCtx.fillStyle = getNodeColor(node, state.colorMode);
      graphCtx.beginPath();
      graphCtx.arc(nx, ny, radius, 0, 2 * Math.PI);
      graphCtx.fill();

      // Halo for express nodes
      if (node.level >= 1) {
        graphCtx.strokeStyle = node.level >= 2 ? "rgba(251, 191, 36, 0.5)" : "rgba(56, 189, 248, 0.4)";
        graphCtx.lineWidth = 1;
        graphCtx.stroke();
      }
    }

    graphCtx.restore();
  }

  // Canvas Interactions: Zoom, Pan & Hover Tooltip
  function initCanvasEvents() {
    if (!graphCanvas) return;

    graphCanvas.addEventListener("mousedown", (e) => {
      state.zoom.isDragging = true;
      state.zoom.startX = e.clientX - state.zoom.x;
      state.zoom.startY = e.clientY - state.zoom.y;
    });

    window.addEventListener("mouseup", () => {
      state.zoom.isDragging = false;
    });

    graphCanvas.addEventListener("mousemove", (e) => {
      if (state.zoom.isDragging) {
        state.zoom.x = e.clientX - state.zoom.startX;
        state.zoom.y = e.clientY - state.zoom.startY;
        renderGraph();
        return;
      }

      // Handle Node Hover
      if (!state.graphData || !state.graphData.nodes) return;
      const rect = graphCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const width = graphCanvas.width;
      const height = graphCanvas.height;
      const scaleFactor = Math.min(width, height) * 0.42;

      const canvasX = (mouseX - (width / 2 + state.zoom.x)) / state.zoom.scale;
      const canvasY = (mouseY - (height / 2 + state.zoom.y)) / state.zoom.scale;

      let foundNode = null;
      let minDistance = 8.0;

      for (const node of state.graphData.nodes) {
        const nx = node.x * scaleFactor;
        const ny = node.y * scaleFactor;
        const d = Math.hypot(canvasX - nx, canvasY - ny);
        if (d < minDistance) {
          minDistance = d;
          foundNode = node;
        }
      }

      if (foundNode && tooltip) {
        tooltip.classList.remove("hidden");
        tooltip.style.left = `${mouseX + 14}px`;
        tooltip.style.top = `${mouseY + 14}px`;
        tooltip.innerHTML = `
          <strong>Vector #${foundNode.id}</strong><br>
          <span style="color:#06b6d4">LID:</span> ${foundNode.lid.toFixed(2)}<br>
          <span style="color:#10b981">Density ($D_k$):</span> ${foundNode.density.toFixed(2)}<br>
          <span style="color:#6366f1">Degree ($M_i$):</span> ${foundNode.degree}<br>
          <span style="color:#fbbf24">Max Layer ($L_i$):</span> ${foundNode.level}<br>
          <span style="color:#f43f5e">Difficulty:</span> ${foundNode.score.toFixed(2)}
        `;
      } else if (tooltip) {
        tooltip.classList.add("hidden");
      }
    });

    graphCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
      state.zoom.scale = Math.max(0.4, Math.min(6.0, state.zoom.scale * zoomFactor));
      renderGraph();
    });

    // Reset Zoom Button
    const btnResetZoom = document.getElementById("btn-reset-zoom");
    if (btnResetZoom) {
      btnResetZoom.addEventListener("click", () => {
        state.zoom = { x: 0, y: 0, scale: 1, isDragging: false, startX: 0, startY: 0 };
        renderGraph();
      });
    }

    // Color Mode Change
    const colorModeSelect = document.getElementById("color-mode");
    if (colorModeSelect) {
      colorModeSelect.addEventListener("change", (e) => {
        state.colorMode = e.target.value;
        updateLegendBounds();
        renderGraph();
      });
    }

    // Edge Toggles
    const toggleL0 = document.getElementById("toggle-layer0-edges");
    const toggleL1 = document.getElementById("toggle-layer1-edges");
    if (toggleL0) toggleL0.addEventListener("change", renderGraph);
    if (toggleL1) toggleL1.addEventListener("change", renderGraph);

    // Sensitivity Slider Listener
    const inputSens = document.getElementById("input-sensitivity");
    if (inputSens) {
      inputSens.addEventListener("input", (e) => {
        document.getElementById("val-sensitivity").textContent = e.target.value;
      });
    }
  }

  // Build & Rebuild Indices via API
  async function handleBuildIndices() {
    const policyType = document.getElementById("policy-type")?.value || "continuous";
    const mBase = parseInt(document.getElementById("input-m-base")?.value || 16);
    const mMin = parseInt(document.getElementById("input-m-min")?.value || 8);
    const mMax = parseInt(document.getElementById("input-m-max")?.value || 28);
    const sens = parseFloat(document.getElementById("input-sensitivity")?.value || 0.5);

    const btn = document.getElementById("btn-rebuild-indices");
    if (btn) btn.textContent = "Building...";

    try {
      const res = await fetch("/api/index/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          policy_type: policyType,
          m_base: mBase,
          m_min: mMin,
          m_max: mMax,
          ef_construction_base: 120,
          sensitivity: sens
        })
      });

      const data = await res.json();
      await fetchStatus();
      await fetchGraphProjection();
      if (btn) btn.textContent = "Build & Profile Graph";
      alert(`Indices built successfully!\nMemory Saved: ${data.comparison.edge_savings_pct}%\nBuild Speedup: ${data.comparison.build_speedup_pct}%`);
    } catch (err) {
      console.error("Build index error:", err);
      if (btn) btn.textContent = "Build & Profile Graph";
    }
  }

  document.getElementById("btn-rebuild-indices")?.addEventListener("click", handleBuildIndices);
  document.getElementById("btn-quick-build")?.addEventListener("click", handleBuildIndices);

  // Generate Dataset Endpoint
  document.getElementById("btn-generate-dataset")?.addEventListener("click", async () => {
    const dsName = document.getElementById("dataset-select")?.value || "multi_manifold";
    const samples = parseInt(document.getElementById("input-samples")?.value || 1500);
    const dim = parseInt(document.getElementById("input-dim")?.value || 32);

    const btn = document.getElementById("btn-generate-dataset");
    if (btn) btn.textContent = "Generating...";

    try {
      await fetch("/api/dataset/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: dsName, n_samples: samples, dim: dim, n_queries: 50 })
      });
      await handleBuildIndices();
      if (btn) btn.textContent = "Generate Dataset";
    } catch (err) {
      console.error(err);
      if (btn) btn.textContent = "Generate Dataset";
    }
  });

  // ==========================================================================
  // TAB 3: Search Simulator Logic & Animation
  // ==========================================================================
  async function runSearchSimulation() {
    const qIdx = parseInt(document.getElementById("sim-query-idx")?.value || 0);
    const k = parseInt(document.getElementById("sim-k")?.value || 10);
    const ef = parseInt(document.getElementById("sim-ef")?.value || 40);

    const btn = document.getElementById("btn-run-sim-search");
    if (btn) btn.textContent = "Tracing...";

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_index: qIdx, k: k, ef_search: ef })
      });

      const data = await res.json();
      state.simTrace = data;
      state.simStepIndex = 0;

      // Update metrics
      document.getElementById("sim-stock-recall").textContent = `${(data.stock.recall * 100).toFixed(0)}%`;
      document.getElementById("sim-adapt-recall").textContent = `${(data.adaptive.recall * 100).toFixed(0)}%`;
      document.getElementById("sim-stock-evals").textContent = data.stock.trace.total_dist_evals || 0;
      document.getElementById("sim-adapt-evals").textContent = data.adaptive.trace.total_dist_evals || 0;

      populateTraceTable(data.adaptive.trace.steps || []);
      renderSimulationStep();

      if (btn) btn.textContent = "Execute Search Trace";
    } catch (err) {
      console.error("Search trace error:", err);
      if (btn) btn.textContent = "Execute Search Trace";
    }
  }

  function populateTraceTable(steps) {
    const tbody = document.getElementById("trace-table-body");
    if (!tbody) return;

    if (!steps || steps.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No trace steps recorded.</td></tr>`;
      return;
    }

    tbody.innerHTML = steps.slice(0, 50).map((s, idx) => `
      <tr>
        <td>#${idx + 1}</td>
        <td><span class="badge ${s.action === 'enter' ? 'cyan-badge' : (s.action === 'explore' ? 'indigo-badge' : 'emerald-badge')}">${s.action}</span></td>
        <td>Layer ${s.layer}</td>
        <td>Node ${s.node !== undefined ? s.node : (s.from || 0)}</td>
        <td>${s.dist ? s.dist.toFixed(4) : '--'}</td>
        <td>Active Search Queue</td>
      </tr>
    `).join("");
  }

  function renderSimulationStep() {
    if (!simCtx || !state.simTrace || !state.graphData) return;

    const width = simCanvas.width;
    const height = simCanvas.height;
    simCtx.clearRect(0, 0, width, height);

    const scaleFactor = Math.min(width, height) * 0.42;

    simCtx.save();
    simCtx.translate(width / 2, height / 2);

    // Draw background nodes faintly
    for (const node of state.graphData.nodes) {
      const nx = node.x * scaleFactor;
      const ny = node.y * scaleFactor;
      simCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
      simCtx.beginPath();
      simCtx.arc(nx, ny, 2, 0, 2 * Math.PI);
      simCtx.fill();
    }

    const steps = state.simTrace.adaptive.trace.steps || [];
    const maxStep = Math.min(state.simStepIndex, steps.length - 1);

    document.getElementById("step-counter").textContent = `Step ${maxStep + 1} / ${steps.length}`;

    // Draw path of visited nodes up to current step
    simCtx.strokeStyle = "rgba(6, 182, 212, 0.8)";
    simCtx.lineWidth = 2.0;
    simCtx.beginPath();

    for (let i = 0; i <= maxStep; i++) {
      const step = steps[i];
      const targetNode = state.graphData.nodes[step.node];
      if (!targetNode) continue;

      const px = targetNode.x * scaleFactor;
      const py = targetNode.y * scaleFactor;

      if (i === 0) {
        simCtx.moveTo(px, py);
      } else {
        simCtx.lineTo(px, py);
      }

      // Highlight visited node
      simCtx.fillStyle = step.layer > 0 ? "#fbbf24" : "#06b6d4";
      simCtx.beginPath();
      simCtx.arc(px, py, step.layer > 0 ? 5 : 3.5, 0, 2 * Math.PI);
      simCtx.fill();
    }
    simCtx.stroke();

    // Highlight Final Top-K Hits if finished
    if (maxStep >= steps.length - 1) {
      for (const hit of state.simTrace.adaptive.results) {
        const hNode = state.graphData.nodes[hit.id];
        if (hNode) {
          const hx = hNode.x * scaleFactor;
          const hy = hNode.y * scaleFactor;
          simCtx.fillStyle = "#10b981";
          simCtx.beginPath();
          simCtx.arc(hx, hy, 6, 0, 2 * Math.PI);
          simCtx.fill();
          simCtx.strokeStyle = "#fff";
          simCtx.lineWidth = 1.5;
          simCtx.stroke();
        }
      }
    }

    simCtx.restore();
  }

  document.getElementById("btn-run-sim-search")?.addEventListener("click", runSearchSimulation);

  document.getElementById("btn-step-next")?.addEventListener("click", () => {
    if (!state.simTrace) return;
    const steps = state.simTrace.adaptive.trace.steps || [];
    if (state.simStepIndex < steps.length - 1) {
      state.simStepIndex++;
      renderSimulationStep();
    }
  });

  document.getElementById("btn-step-prev")?.addEventListener("click", () => {
    if (!state.simTrace) return;
    if (state.simStepIndex > 0) {
      state.simStepIndex--;
      renderSimulationStep();
    }
  });

  document.getElementById("btn-play-animation")?.addEventListener("click", () => {
    if (!state.simTrace) return;
    const steps = state.simTrace.adaptive.trace.steps || [];
    if (state.simPlaying) {
      clearInterval(state.simTimer);
      state.simPlaying = false;
      document.getElementById("btn-play-animation").textContent = "► Auto Play";
    } else {
      state.simPlaying = true;
      document.getElementById("btn-play-animation").textContent = "❚❚ Pause";
      state.simTimer = setInterval(() => {
        if (state.simStepIndex < steps.length - 1) {
          state.simStepIndex++;
          renderSimulationStep();
        } else {
          clearInterval(state.simTimer);
          state.simPlaying = false;
          document.getElementById("btn-play-animation").textContent = "► Auto Play";
        }
      }, 70);
    }
  });

  // ==========================================================================
  // TAB 4: Benchmark Studio Logic & Chart.js
  // ==========================================================================
  async function runFullBenchmark() {
    const efSearch = parseInt(document.getElementById("bench-ef-search")?.value || 50);
    const btn = document.getElementById("btn-run-full-benchmark");
    if (btn) btn.textContent = "Running Benchmark Suite...";

    try {
      const res = await fetch("/api/benchmark/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      const data = await res.json();
      state.benchmarkResults = data.results;

      populateBenchmarkTable(data.results);
      renderBenchmarkCharts(data.results);

      if (btn) btn.textContent = "Run Full Benchmark Suite";
    } catch (err) {
      console.error("Benchmark error:", err);
      if (btn) btn.textContent = "Run Full Benchmark Suite";
    }
  }

  function populateBenchmarkTable(results) {
    const tbody = document.getElementById("benchmark-table-body");
    if (!tbody) return;

    tbody.innerHTML = results.map(r => `
      <tr>
        <td><strong>${r.index_name}</strong></td>
        <td>${r.build_time_s.toFixed(3)}s</td>
        <td>${r.total_edges.toLocaleString()}</td>
        <td><span class="badge ${r.memory_savings_pct > 0 ? 'emerald-badge' : ''}">${r.memory_savings_pct > 0 ? '-' + r.memory_savings_pct.toFixed(1) + '%' : 'Baseline'}</span></td>
        <td>${(r.recall_10 * 100).toFixed(1)}%</td>
        <td>${(r.recall_100 * 100).toFixed(1)}%</td>
        <td><strong>${r.qps.toFixed(1)}</strong></td>
        <td>${r.latency_p95_ms.toFixed(2)} ms</td>
      </tr>
    `).join("");
  }

  function renderBenchmarkCharts(results) {
    if (typeof Chart === "undefined") return;

    const labels = results.map(r => r.index_name.replace(" (Combined Continuous)", "").replace(" (Quantile Tiered)", " (Quantile)"));
    const memorySavings = results.map(r => r.memory_savings_pct);
    const recall10 = results.map(r => r.recall_10 * 100);
    const qpsValues = results.map(r => r.qps);
    const latencies = results.map(r => r.latency_p95_ms);

    // 1. Memory Savings vs Recall Chart
    const ctxMem = document.getElementById("chart-memory-recall")?.getContext("2d");
    if (ctxMem) {
      if (state.chartMemory) state.chartMemory.destroy();
      state.chartMemory = new Chart(ctxMem, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Memory Saved (%)",
              data: memorySavings,
              backgroundColor: "rgba(16, 185, 129, 0.7)",
              borderColor: "#10b981",
              borderWidth: 1,
              yAxisID: "y"
            },
            {
              label: "Recall@10 (%)",
              data: recall10,
              type: "line",
              borderColor: "#06b6d4",
              backgroundColor: "#06b6d4",
              borderWidth: 2,
              pointRadius: 5,
              yAxisID: "y1"
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#94a3b8" } } },
          scales: {
            x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y: { position: "left", title: { display: true, text: "Savings (%)", color: "#10b981" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y1: { position: "right", title: { display: true, text: "Recall@10 (%)", color: "#06b6d4" }, min: 80, max: 100, ticks: { color: "#94a3b8" }, grid: { drawOnChartArea: false } }
          }
        }
      });
    }

    // 2. QPS vs Latency Chart
    const ctxQps = document.getElementById("chart-qps-latency")?.getContext("2d");
    if (ctxQps) {
      if (state.chartQps) state.chartQps.destroy();
      state.chartQps = new Chart(ctxQps, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Throughput (QPS)",
              data: qpsValues,
              backgroundColor: "rgba(99, 102, 241, 0.7)",
              borderColor: "#6366f1",
              borderWidth: 1,
              yAxisID: "y"
            },
            {
              label: "p95 Latency (ms)",
              data: latencies,
              type: "line",
              borderColor: "#f43f5e",
              backgroundColor: "#f43f5e",
              borderWidth: 2,
              pointRadius: 5,
              yAxisID: "y1"
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#94a3b8" } } },
          scales: {
            x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y: { position: "left", title: { display: true, text: "QPS", color: "#6366f1" }, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y1: { position: "right", title: { display: true, text: "p95 Latency (ms)", color: "#f43f5e" }, ticks: { color: "#94a3b8" }, grid: { drawOnChartArea: false } }
          }
        }
      });
    }
  }

  document.getElementById("btn-run-full-benchmark")?.addEventListener("click", runFullBenchmark);

  // ==========================================================================
  // TAB 5: Semantic Document Search Logic
  // ==========================================================================
  async function handleSemanticSearch() {
    const query = document.getElementById("semantic-query-input")?.value || "";
    if (!query) return;

    const btn = document.getElementById("btn-semantic-search");
    if (btn) btn.textContent = "Searching...";

    try {
      const res = await fetch("/api/semantic/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, k: 4 })
      });

      const data = await res.json();

      document.getElementById("badge-stock-semantic-edges").textContent = `Edges: ${data.stock.total_edges}`;
      document.getElementById("badge-adapt-semantic-savings").textContent = `Saved ${data.comparison.edge_savings_pct}% Edges`;

      // Render Stock Hits
      const stockHitsEl = document.getElementById("stock-semantic-hits");
      stockHitsEl.innerHTML = data.stock.hits.map(h => `
        <div class="hit-card">
          <div class="hit-header">
            <span class="hit-title">${h.title}</span>
            <span class="hit-sim">Cosine: ${h.similarity.toFixed(3)}</span>
          </div>
          <div class="hit-cat">${h.category}</div>
          <p class="hit-content">${h.content}</p>
        </div>
      `).join("");

      // Render Adaptive Hits
      const adaptHitsEl = document.getElementById("adapt-semantic-hits");
      adaptHitsEl.innerHTML = data.adaptive.hits.map(h => `
        <div class="hit-card">
          <div class="hit-header">
            <span class="hit-title">${h.title}</span>
            <span class="hit-sim">Cosine: ${h.similarity.toFixed(3)}</span>
          </div>
          <div class="hit-cat">${h.category}</div>
          <p class="hit-content">${h.content}</p>
        </div>
      `).join("");

      if (btn) btn.textContent = "Search Vectors";
    } catch (err) {
      console.error("Semantic search error:", err);
      if (btn) btn.textContent = "Search Vectors";
    }
  }

  document.getElementById("btn-semantic-search")?.addEventListener("click", handleSemanticSearch);

  // Add Document Endpoint
  document.getElementById("btn-add-doc")?.addEventListener("click", async () => {
    const title = document.getElementById("new-doc-title")?.value;
    const cat = document.getElementById("new-doc-category")?.value;
    const content = document.getElementById("new-doc-content")?.value;

    if (!title || !content) {
      alert("Please enter title and content");
      return;
    }

    try {
      const res = await fetch("/api/semantic/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title, category: cat || "General", content: content })
      });
      const data = await res.json();
      alert(`Document added! Total corpus size: ${data.total_docs}`);
      document.getElementById("new-doc-title").value = "";
      document.getElementById("new-doc-content").value = "";
    } catch (err) {
      console.error(err);
    }
  });

  // ==========================================================================
  // TAB 6: Export CSV & Project Report
  // ==========================================================================
  document.getElementById("btn-export-csv")?.addEventListener("click", () => {
    if (state.benchmarkResults.length === 0) {
      alert("Please run the benchmark suite first to generate results.");
      return;
    }

    let csv = "Index Name,Build Time (s),Total Edges,Avg Edges/Node,Memory Saved (%),Recall@10,Recall@100,QPS,p95 Latency (ms)\n";
    state.benchmarkResults.forEach(r => {
      csv += `"${r.index_name}",${r.build_time_s},${r.total_edges},${r.avg_edges_per_node},${r.memory_savings_pct},${r.recall_10},${r.recall_100},${r.qps},${r.latency_p95_ms}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.setAttribute("hidden", "");
    a.setAttribute("href", url);
    a.setAttribute("download", "adaptivevec_benchmark_results.csv");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

});
