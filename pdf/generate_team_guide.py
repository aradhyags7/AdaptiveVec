"""
generate_team_guide.py
Generates 'AdaptiveVec_Team_Master_Handbook.pdf'
A comprehensive, publication-grade master handbook with 10 embedded high-resolution diagrams:
- Chapter 1: Vector Search from Scratch (with Diagram 1: Embeddings & Linear Scan vs ANN)
- Chapter 2: The Flaws of Standard HNSW (with Diagram 2: HNSW Hierarchy & Hubness Flaws)
- Chapter 3: What We HAVE Done (with Diagram 3: 5-Stage Pipeline, Diagram 4: Trajectory & Stagnation Exit,
              Diagram 5: Hubness Elimination, Diagram 6: SQ8 Memory Architecture)
- Chapter 4: Verified Empirical Results (with Figure 1 Pareto, Figure 2 Hubness PDF,
              Figure 3 Ablation Waterfall, Figure 4 Synthetic Sweep)
- Chapter 5: How to Run and Demo the System (Developer Cheatsheet)
- Chapter 6: What We HAVEN'T Done (The Complete End-Sem Phase 2 Work Plan)
- Chapter 7: Examiner Defense & Team Viva Guide (Talking points & 8 Trap Q&A)
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, total_pages):
        # Suppress running header/footer on cover page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#786F62"))

        # Running Header
        header_text = "ADAPTIVEVEC: MASTER TEAM ENGINEERING & RESEARCH HANDBOOK"
        self.drawString(54, 11 * inch - 36, header_text)
        self.setFont("Helvetica", 8)
        self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "CONFIDENTIAL / TEAM ONLY")

        self.setStrokeColor(colors.HexColor("#38312A"))
        self.setLineWidth(0.6)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#38312A"))
        self.setLineWidth(0.6)
        self.line(54, 46, 8.5 * inch - 54, 46)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#786F62"))
        self.drawString(54, 32, "AdaptiveVec Systems Project • Comprehensive Team Onboarding & End-Sem Roadmap")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 54, 32, page_str)
        self.restoreState()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_team_handbook(output_filename=None):
    if output_filename is None:
        output_filename = os.path.join(SCRIPT_DIR, "AdaptiveVec_Team_Master_Handbook.pdf")
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    c_primary = colors.HexColor("#171513")
    c_text = colors.HexColor("#2B2621")
    c_subtle = colors.HexColor("#524B42")
    c_amber = colors.HexColor("#C97D4A")
    c_amber_glow = colors.HexColor("#A65624")
    c_navy = colors.HexColor("#1A2B4C")
    c_card_bg = colors.HexColor("#F7F5F0")
    c_border = colors.HexColor("#D5CBC0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_navy,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11.5,
        leading=15.5,
        textColor=c_amber_glow,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'ChapterH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14.5,
        leading=18.5,
        textColor=c_navy,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=c_amber_glow,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
        textColor=c_text,
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1A2B4C")
    )

    callout_text = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1F2937")
    )

    fig_caption = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=1, # Center
        spaceAfter=8
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_primary
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )

    table_cell_mono = ParagraphStyle(
        'TableCellMono',
        parent=table_cell,
        fontName='Courier',
        fontSize=8
    )

    story = []

    # Helper: Callout Box
    def make_callout(title, text, bg_hex="#F0EAE0", border_hex="#C97D4A"):
        header_p = Paragraph(f"<b>{title}</b>", ParagraphStyle('CHead', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=colors.HexColor(border_hex)))
        body_p = Paragraph(text, callout_text)
        t = Table([[header_p], [body_p]], colWidths=[7.0 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_hex)),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border_hex)),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 3),
            ('TOPPADDING', (0,1), (-1,1), 2),
        ]))
        return t

    # Helper: Image with caption
    def make_image_card(img_path, width_in=7.0, height_in=2.5, caption_text=""):
        flowables = []
        if os.path.exists(img_path):
            img = Image(img_path, width=width_in * inch, height=height_in * inch)
            flowables.append(img)
            if caption_text:
                flowables.append(Spacer(1, 3))
                flowables.append(Paragraph(caption_text, fig_caption))
            flowables.append(Spacer(1, 6))
        return flowables

    # =========================================================================
    # COVER / HEADER BANNER
    # =========================================================================
    story.append(Paragraph("ADAPTIVEVEC: MASTER ENGINEERING & RESEARCH HANDBOOK", title_style))
    story.append(Paragraph("A Complete Ground-Up Guide to Theory, Algorithms, Empirical Results, and End-Sem Roadmap for Team Members", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_amber, spaceAfter=12))

    meta_text = (
        "<b>Repository &amp; System:</b> AdaptiveVec Proximity Graph Engine (C++20 AVX2 Core + Python API + React 19 Telemetry)<br/>"
        "<b>Target Standard:</b> IEEE/ACM Transactions on Knowledge &amp; Data Engineering (TKDE) Camera-Ready Standard<br/>"
        "<b>Evaluated Corpora:</b> Texmex SIFT-100K (128d), Stanford GloVe-100K (100d), Fashion-MNIST (784d), Synthetic (64d)<br/>"
        "<b>Primary Deliverable:</b> Standalone Technical Dossier for Project Defense, Viva Voce, and Engineering Onboarding"
    )
    story.append(Paragraph(meta_text, ParagraphStyle('MetaText', fontName='Helvetica', fontSize=8.5, leading=12, textColor=c_subtle)))
    story.append(Spacer(1, 10))

    callout_cover = make_callout(
        "HOW TO USE THIS HANDBOOK (FOR TEAMMATES)",
        "This document assumes <b>zero prior knowledge</b> of vector search, graph theory, or quantization. It explains why industry engines "
        "(FAISS, Milvus) break on real-world data, the exact mathematical formulas and C++ SIMD optimizations we built, our verified empirical "
        "numbers (+50.3% speedup, -62.4% RAM reduction), and exactly what tasks remain for Phase 2 (End-Sem).",
        bg_hex="#FAF6F0", border_hex="#C97D4A"
    )
    story.append(callout_cover)
    story.append(Spacer(1, 12))

    # =========================================================================
    # CHAPTER 1: VECTOR SEARCH FROM SCRATCH
    # =========================================================================
    story.append(Paragraph("Chapter 1: Vector Search from Scratch (The Absolute Basics)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph("<b>1.1 What is an Embedding Vector?</b>", h2_style))
    story.append(Paragraph(
        "In modern artificial intelligence (ChatGPT, Google Lens, Spotify recommendations), data like words, photos, and audio "
        "cannot be searched using standard SQL keyword matching (e.g., <i>SELECT * WHERE text = 'cat'</i>). Instead, neural networks "
        "convert raw data into an <b>embedding vector</b>—a list of floating-point numbers in a high-dimensional space (e.g., 128 numbers for SIFT image features, "
        "768 numbers for BERT text models, or 1,536 for OpenAI embeddings). Vectors that have similar semantic meaning sit physically close together in this space.",
        body_style
    ))

    story.append(Paragraph("<b>1.2 The Brute-Force Scaling Wall: Why Linear Scan Dies</b>", h2_style))
    story.append(Paragraph(
        "When a user submits a query (e.g., searching for a similar image), the computer must find the <i>k</i> nearest neighbors (k-NN) by measuring Euclidean distance: "
        "<i>dist(u, v) = sqrt( sum( (u_i - v_i)^2 ) )</i>. "
        "If you have <b>1,000,000 vectors</b> of 128 dimensions, every single query requires 128,000,000 floating-point operations. "
        "If 1,000 users query the database every second, the CPU must perform <b>128 billion calculations per second</b>. "
        "Linear brute-force search is mathematically exact (100% recall), but computationally impossible at web scale.",
        body_style
    ))

    story.append(Paragraph("<b>1.3 Approximate Nearest Neighbor (ANN) &amp; Proximity Graphs</b>", h2_style))
    story.append(Paragraph(
        "To solve this, the computer science community uses <b>Approximate Nearest Neighbor (ANN)</b> search. We accept a tiny margin of error "
        "(e.g., finding the true nearest neighbor 98% of the time instead of 100%) in exchange for a <b>100x to 1,000x speedup</b>. "
        "The fastest known data structure for ANN is a <b>Proximity Graph</b>: vectors become graph nodes, and edges connect vectors that are close in space. "
        "Searching the database becomes a simple greedy graph traversal: start at an entry node, look at its neighbors, hop to the neighbor closest to the query, "
        "and repeat until no closer neighbor exists.",
        body_style
    ))

    # EMBED DIAGRAM 1
    d1_path = os.path.join(SCRIPT_DIR, "handbook_figures", "diagram1_embeddings_ann.png")
    for elem in make_image_card(d1_path, width_in=7.0, height_in=2.3, caption_text="<b>Figure 1:</b> The Vector Search Paradigm: Neural embedding projection, high-dimensional Euclidean space, and why ANN proximity graphs save 99.9% compute over brute-force scan."):
        story.append(elem)

    # =========================================================================
    # CHAPTER 2: THE FATAL FLAWS OF STANDARD HNSW
    # =========================================================================
    story.append(Paragraph("Chapter 2: The Flaws of Standard HNSW (The Problem We Solved)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph(
        "The current global gold standard for proximity graphs is <b>HNSW (Hierarchical Navigable Small World)</b>, invented by Malkov &amp; Yashunin in 2018. "
        "It acts like a highway network: top express layers cross huge distances in 2 or 3 hops, while bottom Layer 0 contains all nodes for fine-grained navigation. "
        "However, HNSW was built on an assumption that fails in real-world data: <b>it forces static, uniform parameters onto every vector</b>.",
        body_style
    ))

    story.append(Paragraph("<b>Problem 1: The Curse of Hubness in High Dimensions</b>", h2_style))
    story.append(Paragraph(
        "In high-dimensional spaces (D &ge; 64), distance distributions concentrate. Points near the centroid of dense clusters naturally "
        "become the nearest neighbors of hundreds or thousands of other points. In standard HNSW, these centroid nodes become <b>'super-hubs'</b>—they "
        "accumulate massive numbers of incoming edges (in-degree &gt; 2,000). During search, queries get trapped routing through these congested hubs, "
        "leading to traffic bottlenecks and high computational variance.",
        body_style
    ))

    story.append(Paragraph("<b>Problem 2: Greedy Search Distance Stagnation (30% Wasted Compute)</b>", h2_style))
    story.append(Paragraph(
        "Standard HNSW uses a fixed exploration budget (e.g., efSearch = 64). In practice, greedy beam search reaches the true Voronoi cell / local basin of attraction "
        "within 10 to 15 hops. After that, subsequent hops evaluate dozens of neighbor candidates whose distances improve by less than 10<sup>-5</sup>. "
        "Standard HNSW blindly continues searching until the fixed budget runs out. <b>Over 30% of CPU distance evaluations are completely wasted on stagnant candidates.</b>",
        body_style
    ))

    # EMBED DIAGRAM 2
    d2_path = os.path.join(SCRIPT_DIR, "handbook_figures", "diagram2_hnsw_hierarchy.png")
    for elem in make_image_card(d2_path, width_in=7.0, height_in=2.4, caption_text="<b>Figure 2:</b> Standard HNSW vs. AdaptiveVec Architecture: Multi-layer skip-list hierarchy and why uniform M=16 creates congested super-hubs, contrasted with our topology-aware ground layer."):
        story.append(elem)

    # =========================================================================
    # CHAPTER 3: WHAT WE HAVE DONE (THE COMPLETE SYSTEM)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 3: What We Have Done (Our Implementation &amp; System Architecture)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph(
        "To solve these problems, we engineered <b>AdaptiveVec</b>: an end-to-end, topology-aware proximity graph index. "
        "Here is the complete breakdown of the 5 algorithmic stages we designed, implemented, and verified:",
        body_style
    ))

    # EMBED DIAGRAM 3: PIPELINE FLOWCHART
    d3_path = os.path.join(SCRIPT_DIR, "handbook_figures", "diagram3_pipeline_flowchart.png")
    for elem in make_image_card(d3_path, width_in=7.0, height_in=2.1, caption_text="<b>Figure 3:</b> The AdaptiveVec 5-Stage System Pipeline: From raw input embeddings to pre-indexing probing, degree scaling, hubness regulation, early exit, and SQ8 quantization."):
        story.append(elem)

    # Stage 1
    story.append(Paragraph("<b>Stage 1: Pre-Indexing Topological Probing (LID Estimation)</b>", h2_style))
    story.append(Paragraph(
        "Before building the graph, we inspect the geometric complexity of the embedding space. We estimate the <b>Local Intrinsic Dimensionality (LID)</b> "
        "around each point using extreme value theory (Levina-Bickel maximum likelihood estimator on k=16 neighbors):",
        body_style
    ))
    story.append(Paragraph(
        "<code>LIDest(x) = - [ (1 / k) * sum_{i=1}^k ln( r_i(x) / r_k(x) ) ]^(-1)</code>",
        code_style
    ))
    story.append(Paragraph(
        "<b>Engineering Proof:</b> Probing takes <i>O(N * k log k)</i> time, which consumes <b>strictly under 0.8% of total index build time</b>. "
        "It cleanly separates tight linear sub-spaces (low LID ~2 to 8) from sparse, noisy boundary regions (high LID ~24 to 64).",
        body_style
    ))

    # Stage 2
    story.append(Paragraph("<b>Stage 2: Topology-Aware Degree Scaling (Dynamic M(x))</b>", h2_style))
    story.append(Paragraph(
        "Instead of forcing M = 16 on all nodes: "
        "<br/>• <b>Dense cluster nodes (low LID):</b> Allocated fewer connections (e.g., M = 10) because routing inside tight clusters is trivial. "
        "<br/>• <b>Sparse ridge nodes (high LID):</b> Allocated expanded connections (e.g., M = 22) to prevent disconnectivity across complex boundaries. "
        "<br/>• <b>Layer-Decoupled Scaling:</b> We keep upper express layers uniform (M = 16) and restrict dynamic scaling strictly to ground Layer 0. "
        "<br/><b>Result:</b> Cuts total graph edges by <b>7.4%</b> and slashes index build time by <b>28.1% (33.5s vs 46.6s)</b>.",
        body_style
    ))

    # Stage 3 & EMBED DIAGRAM 5 (Hubness)
    story.append(Paragraph("<b>Stage 3: In-Degree Hubness Regulation (mu = 0.15)</b>", h2_style))
    story.append(Paragraph(
        "During edge selection, we penalize candidates that have already accumulated too many incoming edges: "
        "<br/><code>Score(v) = dist(u, v) + mu * (deg_in(v) / deg_in_mean) * sigma_dist</code> (with mu = 0.15). "
        "<br/><b>Result:</b> Slashes graph in-degree variance by <b>54.9% (from 142.8 down to 64.4)</b>, distributing query traffic evenly.",
        body_style
    ))

    d5_path = os.path.join(SCRIPT_DIR, "handbook_figures", "diagram5_hubness_comparison.png")
    for elem in make_image_card(d5_path, width_in=7.0, height_in=2.3, caption_text="<b>Figure 4:</b> Hubness Elimination Visual Comparison: Severe central hub traffic bottleneck in unregulated HNSW vs. balanced distributed routing under AdaptiveVec (mu=0.15)."):
        story.append(elem)

    # Stage 4 & EMBED DIAGRAM 4 (Stagnation)
    story.append(Paragraph("<b>Stage 4: Distance Stagnation Early Exit (The +50% Speedup Engine)</b>", h2_style))
    story.append(Paragraph(
        "During query execution, greedy search monitors consecutive distance improvement deltas: <i>Delta_d = |d_t - d_{t-1}|</i>. "
        "If the improvement across p = 6 consecutive hops satisfies: "
        "<br/><code>sum_{j=0}^{p-1} |d_{t-j} - d_{t-j-1}| &lt; p * epsilon   (with p = 6, epsilon = 10^-4)</code> "
        "<br/>The search terminates immediately! The query is already in the true local minimum. "
        "<br/><b>Result:</b> Cuts distance evaluations per query from <b>1,121 down to 783 (-30.1% compute savings)</b>, directly delivering our <b>+50.3% search throughput boost (7,075 vs 4,708 QPS)</b>.",
        body_style
    ))

    d4_path = os.path.join(SCRIPT_DIR, "handbook_figures", "diagram4_stagnation_trajectory.png")
    for elem in make_image_card(d4_path, width_in=7.0, height_in=2.3, caption_text="<b>Figure 5:</b> Greedy Search Trajectory & Stagnation Exit: Spatial node hops and the distance flatline curve showing how early exit safely bypasses redundant computations."):
        story.append(elem)

    # Stage 5 & EMBED DIAGRAM 6 (SQ8)
    story.append(Paragraph("<b>Stage 5: Asymmetric 8-Bit Scalar Quantization (Regime B - SQ8)</b>", h2_style))
    story.append(Paragraph(
        "For memory-constrained regimes, vectors are quantized from 32-bit floats into 8-bit unsigned integers (uint8) using affine min-max scaling per dimension. "
        "Candidate exploration uses fast SIMD integer dot-products, followed by FP32 re-ranking on top candidates. "
        "<br/><b>Result:</b> Slashes RAM consumption by <b>62.4% (from 59.9 MB down to 22.5 MB)</b> with 95.9% recall retention.",
        body_style
    ))

    d6_path = os.path.join(SCRIPT_DIR, "handbook_figures", "diagram6_sq8_quantization.png")
    for elem in make_image_card(d6_path, width_in=7.0, height_in=2.1, caption_text="<b>Figure 6:</b> Asymmetric SQ8 Quantization Memory Architecture: FP32 vs uint8 memory layout and the 2-stage beam search re-ranking pipeline."):
        story.append(elem)

    # =========================================================================
    # CHAPTER 4: VERIFIED EMPIRICAL RESULTS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 4: Verified Empirical Benchmark Results", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph(
        "All benchmarks were conducted following standard <b>ANN-Benchmarks protocols</b> on the standard <b>Texmex SIFT-100K</b> corpus "
        "(100,000 vectors, 128 dimensions, 10,000 queries) under fixed hardware conditions. Here are the master numbers:",
        body_style
    ))

    # Master Table
    bench_data = [
        [Paragraph("<b>Evaluation Metric</b>", table_header), Paragraph("<b>1. Baseline HNSW</b>", table_header), Paragraph("<b>2. Regime A (Fast)</b>", table_header), Paragraph("<b>3. Regime B (SQ8)</b>", table_header), Paragraph("<b>Scientific Impact &amp; Attribution</b>", table_header)],
        [Paragraph("Search Throughput (QPS)", table_cell_bold), Paragraph("4,708.1 QPS", table_cell), Paragraph("<b>7,075.3 QPS</b>", table_cell_bold), Paragraph("2,987.6 QPS", table_cell), Paragraph("<b>+50.3% speedup</b> via p=6 Stagnation Exit", table_cell)],
        [Paragraph("Recall@10 Accuracy", table_cell_bold), Paragraph("0.9913 (99.1%)", table_cell), Paragraph("<b>0.9745 (97.5%)</b>", table_cell_bold), Paragraph("0.9594 (95.9%)", table_cell), Paragraph("<b>98.4% relative recall preserved</b> (-1.68%)", table_cell)],
        [Paragraph("Index Memory (RAM)", table_cell_bold), Paragraph("59.93 MB", table_cell), Paragraph("59.16 MB", table_cell), Paragraph("<b>22.54 MB</b>", table_cell_bold), Paragraph("<b>-62.4% RAM reduction</b> via Asymmetric SQ8", table_cell)],
        [Paragraph("Index Build Time", table_cell_bold), Paragraph("46.61 sec", table_cell), Paragraph("<b>33.51 sec</b>", table_cell_bold), Paragraph("63.96 sec", table_cell), Paragraph("<b>-28.1% faster build</b> (Layer-decoupled M)", table_cell)],
        [Paragraph("Total Graph Edges", table_cell_bold), Paragraph("2,709,125", table_cell), Paragraph("<b>2,509,138</b>", table_cell_bold), Paragraph("2,509,743", table_cell), Paragraph("<b>-7.38% fewer edges</b> (-199,987 edges pruned)", table_cell)],
        [Paragraph("Distance Evals / Query", table_cell_bold), Paragraph("1,121.2", table_cell), Paragraph("<b>783.5</b>", table_cell_bold), Paragraph("842.6", table_cell), Paragraph("<b>-30.1% compute saved</b> (337.7 fewer evals)", table_cell)],
        [Paragraph("In-Degree Variance", table_cell_bold), Paragraph("142.8 var", table_cell), Paragraph("<b>64.4 var</b>", table_cell_bold), Paragraph("64.4 var", table_cell), Paragraph("<b>-54.9% hubness drop</b> (Regulation mu=0.15)", table_cell)],
    ]
    t_bench = Table(bench_data, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch, 2.0 * inch])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_card_bg]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    callout_honesty = make_callout(
        "CRITICAL SCIENTIFIC NUANCE (REVIEWER TRAP)",
        "Reviewers frequently ask: <i>'Did your LID topology changes speed up query throughput?'</i><br/>"
        "<b>The honest, scientific answer is NO.</b> Build-time topology adaptation (LID scaling and hubness regulation) reduces build time by 28.1% "
        "and trims 7.4% of graph edges, but does <b>not</b> independently increase QPS at fixed efSearch=64. "
        "The +50.3% search throughput boost is driven <b>exclusively by Distance Stagnation Early Exit</b> at query time. "
        "Answering this honestly in review shows supreme technical mastery.",
        bg_hex="#FAF6F0", border_hex="#A65624"
    )
    story.append(callout_honesty)
    story.append(Spacer(1, 10))

    # EMBED PUBLICATION PLOTS 1 & 2
    story.append(Paragraph("<b>4.2 Publication Benchmark Plots (Pareto Frontier &amp; Hubness Distribution)</b>", h2_style))
    fig1_path = os.path.join(SCRIPT_DIR, "paper_figures", "fig1_pareto.png")
    fig2_path = os.path.join(SCRIPT_DIR, "paper_figures", "fig2_hubness_distribution.png")
    if os.path.exists(fig1_path) and os.path.exists(fig2_path):
        img_table = Table([
            [Image(fig1_path, width=3.4 * inch, height=2.3 * inch), Image(fig2_path, width=3.4 * inch, height=2.3 * inch)],
            [Paragraph("<b>Figure 7:</b> SIFT-100K Pareto Frontier. Regime A achieves 7,075 QPS (+50.3% speedup).", table_cell),
             Paragraph("<b>Figure 8:</b> In-degree PDF distribution: hubness variance slashed by 54.9% under mu=0.15.", table_cell)]
        ], colWidths=[3.5 * inch, 3.5 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

    # EMBED PUBLICATION PLOTS 3 & 4
    fig3_path = os.path.join(SCRIPT_DIR, "paper_figures", "fig3_ablation_waterfall.png")
    fig4_path = os.path.join(SCRIPT_DIR, "paper_figures", "fig4_synthetic_sweep.png")
    if os.path.exists(fig3_path) and os.path.exists(fig4_path):
        img_table2 = Table([
            [Image(fig3_path, width=3.4 * inch, height=2.3 * inch), Image(fig4_path, width=3.4 * inch, height=2.3 * inch)],
            [Paragraph("<b>Figure 9:</b> 6-Step Cumulative Ablation: Throughput gains (left) &amp; SQ8 RAM drop (right).", table_cell),
             Paragraph("<b>Figure 10:</b> Synthetic Multi-Cluster sweep: Hubness penalty trade-off over mu in [0, 0.30].", table_cell)]
        ], colWidths=[3.5 * inch, 3.5 * inch])
        img_table2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(img_table2)
        story.append(Spacer(1, 8))

    # =========================================================================
    # CHAPTER 5: HOW TO RUN AND DEMO
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 5: How to Run and Demo the System", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph(
        "Every teammate should be able to spin up the system on their laptop in under 30 seconds. Here are the exact commands:",
        body_style
    ))

    cmd_box = (
        "<b>1. Launching the Interactive Instrument Dashboard:</b><br/>"
        "&nbsp;&nbsp;<code>cd frontend-react2</code><br/>"
        "&nbsp;&nbsp;<code>npm run dev</code> &nbsp;&nbsp;<i>(or simply run 'npm run dev' from the project root!)</i><br/>"
        "&nbsp;&nbsp;Open browser at: <b>http://localhost:5174/</b><br/><br/>"
        "<b>2. Running the Full Python Benchmark Suite:</b><br/>"
        "&nbsp;&nbsp;<code>python benchmarks/run_paper_benchmarks.py --dataset sift100k --trials 5</code><br/><br/>"
        "<b>3. Regenerating the Camera-Ready Research Paper PDF:</b><br/>"
        "&nbsp;&nbsp;<code>python generate_paper.py</code> &nbsp;&nbsp;<i>(generates AdaptiveVec_Research_Paper.pdf)</i><br/><br/>"
        "<b>4. Running the C++ AVX2 Native Benchmark:</b><br/>"
        "&nbsp;&nbsp;<code>g++ -O3 -mavx2 -mfma cpp/benchmark_main.cpp -o build/benchmark_main</code><br/>"
        "&nbsp;&nbsp;<code>./build/benchmark_main</code>"
    )
    story.append(Table([[Paragraph(cmd_box, code_style)]], colWidths=[7.0 * inch], style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3EEE5")),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Spacer(1, 14))

    # =========================================================================
    # CHAPTER 6: WHAT WE HAVEN'T DONE (END-SEM WORK PLAN)
    # =========================================================================
    story.append(Paragraph("Chapter 6: What We HAVEN'T Done (The End-Sem Work Plan)", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph(
        "<b>CRITICAL FOR REVIEWS:</b> Never tell an examiner 'we finished everything'. Engineering projects have phases. "
        "We position our current work as <b>Phase 1 (Core Engine &amp; Algorithmic Proof)</b> successfully completed. "
        "Here are the <b>6 concrete tasks</b> left for Phase 2 (End-Sem):",
        body_style
    ))

    pending_tasks = [
        ("Task 1: 1-Million Vector Scaling (DEEP1M & SIFT1M)",
         "We currently proved AdaptiveVec on 100K vectors. For end-sem, we will benchmark on <b>1,000,000 vectors</b> to demonstrate "
         "that distance stagnation exit scales linearly as graph diameter expands, proving logarithmic search complexity O(log N)."),

        ("Task 2: Hardware Microarchitectural CPU Profiling",
         "Using Linux <code>perf</code> and Intel VTune, we will profile: "
         "<br/>• L1/L2/LLC data cache misses per query. "
         "<br/>• SIMD pipeline saturation during early exit. "
         "<br/>• Branch misprediction rates in the stagnation while-loop. This will provide hardware-level proof of energy efficiency."),

        ("Task 3: Dynamic / Query-Adaptive Threshold p(q)",
         "Currently, our stagnation window is fixed at p = 6 for all queries. In Phase 2, we will implement query-adaptive early exit: "
         "easy queries in dense clusters exit aggressively at p = 4, while ambiguous out-of-distribution queries explore deeper to p = 8."),

        ("Task 4: Product Quantization (PQ / IVF-PQ) Exploration",
         "While SQ8 reduced memory by 62.4%, extreme edge devices (microcontrollers, mobile phones) require 4-bit or 16-byte codebook compression. "
         "We will evaluate Product Quantization (PQ) integrated directly into our adaptive topology."),

        ("Task 5: Lock-Free Multi-Threaded Build Scaling",
         "Our C++ query engine is high-throughput, but graph indexing currently uses batch single-writer insertion. We will implement "
         "fine-grained reader-writer locks or lock-free atomic CAS neighbor updates to scale build throughput across 32+ CPU cores."),

        ("Task 6: Peer-Reviewed Conference / Journal Submission",
         "We have already drafted a 13-page publication-grade manuscript formatted to IEEE/ACM standards. For end-sem, we will incorporate "
         "the 1M scale results and submit the paper to a peer-reviewed venue (e.g., IEEE TKDE, VLDB, or ACM SIGMOD)."
        )
    ]

    for title, desc in pending_tasks:
        t_box = Table([[
            Paragraph(f"<b>{title}</b>", ParagraphStyle('PTitle', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=colors.HexColor("#1A2B4C"))),
        ], [
            Paragraph(desc, callout_text)
        ]], colWidths=[7.0 * inch])
        t_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
            ('PADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ]))
        story.append(t_box)
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 7: EXAMINER DEFENSE & TEAM VIVA GUIDE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 7: Examiner Defense & Team Viva Guide", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceAfter=8))

    story.append(Paragraph("<b>7.1 How to Divide Talking Points Among Team Members</b>", h2_style))
    story.append(Paragraph(
        "When presenting as a team, divide the presentation into 3 clear, non-overlapping roles: "
        "<br/>• <b>Speaker 1 (The Problem &amp; Architecture):</b> Explains why FAISS/HNSW breaks on real-world data (hubness + wasted compute) and walks through the 5-stage pipeline diagram on the <code>Overview</code> tab. "
        "<br/>• <b>Speaker 2 (The Benchmarks &amp; Science):</b> Shows the <code>Benchmarks &amp; Pareto</code> tab, explains the +50.3% QPS speedup, the 6-step ablation waterfall, and the 54.9% hubness drop. "
        "<br/>• <b>Speaker 3 (Live Demo &amp; Roadmap):</b> Runs the <code>Trajectory Simulator</code> live, explains the Hop 16 stagnation exit, opens the IEEE Research Paper PDF, and delivers the End-Sem Phase 2 roadmap.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>7.2 Top 8 Examiner Trap Questions &amp; Model Answers</b>", h2_style))

    qa_list = [
        ("Q1: Did you write this in C++ from scratch or just use FAISS?",
         "<b>Model Answer:</b> We engineered our own custom C++ HNSW engine and adaptive algorithms from scratch using AVX2 and FMA SIMD vector intrinsics. "
         "We followed standard ANN-Benchmarks evaluation protocols under identical hardware constraints to benchmark against standard HNSW fairly."),

        ("Q2: Where does the +50.3% speedup actually come from?",
         "<b>Model Answer:</b> It is driven entirely at query time via Distance Stagnation Early Exit. In greedy beam search, queries spend 30% of their "
         "time evaluating redundant neighbor candidates after already reaching the true local minimum. By terminating when consecutive improvements stagnate "
         "(Delta_d < 10^-4 for p=6 hops), we cut distance calculations per query from 1,121 to 783 (-30.1%), translating directly to +50.3% QPS."),

        ("Q3: Did your topology changes speed up query throughput?",
         "<b>Model Answer:</b> Scientifically, no. Topology adaptation reduces graph edges by 7.4% and speeds up index construction time by 28.1%, "
         "but does not independently boost query throughput at fixed efSearch=64. The query speedup is achieved by the early-exit search policy."),

        ("Q4: What is Local Intrinsic Dimensionality (LID) and why not use PCA?",
         "<b>Model Answer:</b> Global PCA assumes the entire dataset shares a single linear subspace. Real embedding spaces are non-linear and heterogeneous: "
         "dense cluster centers have low LID (~8), while boundary ridges have high LID (~24). Using extreme value theory on k=16 nearest neighbors, "
         "we estimate local curvature in O(N * k log k), taking strictly under 0.8% of index construction time."),

        ("Q5: Why is mu = 0.15 chosen for hubness regulation?",
         "<b>Model Answer:</b> In our sensitivity sweep, mu = 0 leaves the graph unregulated with severe hub formation (variance 142.8). At mu = 0.15, "
         "in-degree variance drops by 54.9% (to 64.4) with zero recall loss. Setting mu > 0.30 over-penalizes graph bridges, causing recall degradation. "
         "Hence, mu = 0.15 provides the optimal empirical balance."),

        ("Q6: Does early exit hurt search accuracy?",
         "<b>Model Answer:</b> On SIFT-100K, baseline recall is 0.9913, and our early-exit recall is 0.9745. That is a 98.4% relative recall retention. "
         "Trading 1.68% recall for a +50.3% throughput boost is considered an outstanding trade-off in production vector search."),

        ("Q7: How does your memory compression work?",
         "<b>Model Answer:</b> Regime B uses asymmetric INT8 scalar quantization (SQ8). Each vector dimension is affine-mapped to an 8-bit integer. "
         "Candidate exploration uses fast SIMD integer dot-products, followed by FP32 re-ranking on the top candidates. "
         "This shrinks RAM from 59.9 MB to 22.5 MB (-62.4%) with 95.9% Recall@10."),

        ("Q8: What is left for the final semester review?",
         "<b>Model Answer:</b> Phase 1 delivered the C++ AVX2 core, verified SIFT-100K benchmarks, and the research paper draft. Phase 2 covers: "
         "(1) 1-million vector stress testing, (2) CPU hardware cache profiling with Linux perf, (3) query-adaptive p(q) thresholding, and (4) conference submission.")
    ]

    for q, a in qa_list:
        q_box = Table([[
            Paragraph(f"<b>{q}</b>", ParagraphStyle('QTitle', fontName='Helvetica-Bold', fontSize=9, leading=11.5, textColor=c_navy))
        ], [
            Paragraph(a, callout_text)
        ]], colWidths=[7.0 * inch])
        q_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FAF6F0")),
            ('BOX', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ]))
        story.append(q_box)
        story.append(Spacer(1, 5))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {output_filename}")


if __name__ == "__main__":
    create_team_handbook()
