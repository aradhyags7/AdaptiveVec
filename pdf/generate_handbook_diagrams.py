"""
generate_handbook_diagrams.py
Generates 6 clear, publication-quality conceptual and architectural diagrams
for the AdaptiveVec Master Team Handbook.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "handbook_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set global styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#94a3b8'
plt.rcParams['axes.linewidth'] = 0.8

# =============================================================================
# DIAGRAM 1: Vector Embeddings & The Linear Scan Bottleneck
# =============================================================================
def make_diagram1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300)
    
    # Subplot 1: Unstructured Data to Vector
    ax = axes[0]
    ax.set_title("1. Unstructured Data to Embeddings", fontsize=11, fontweight='bold', color='#1A2B4C', pad=12)
    ax.axis('off')
    
    # Draw boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#F1F5F9', edgecolor='#64748B', linewidth=1)
    embed_props = dict(boxstyle='round,pad=0.5', facecolor='#EFF6FF', edgecolor='#2563EB', linewidth=1.5)
    
    ax.text(0.5, 0.88, "Raw Unstructured Input\n(Text, Image, Audio, Graph)", ha='center', va='center', bbox=box_props, fontsize=9.5)
    ax.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.78), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2563EB'))
    ax.text(0.5, 0.62, "Deep Neural Network\n(CLIP, ResNet, BERT, OpenAI)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF3C7', edgecolor='#D97706', lw=1.2), fontsize=9)
    ax.annotate("", xy=(0.5, 0.42), xytext=(0.5, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2563EB'))
    ax.text(0.5, 0.28, "Dense Embedding Vector\n[0.24, -0.81, 0.45, ... 0.12]\n(D = 128 Dimensions, Float32)", ha='center', va='center', bbox=embed_props, fontsize=9.5, fontweight='bold', color='#1E40AF')
    
    # Subplot 2: Geometric Embedding Space
    ax = axes[1]
    ax.set_title("2. Geometric Embedding Space", fontsize=11, fontweight='bold', color='#1A2B4C', pad=12)
    np.random.seed(42)
    c1 = np.random.randn(30, 2) * 0.4 + [-1.2, 0.8]
    c2 = np.random.randn(30, 2) * 0.4 + [1.2, -0.6]
    c3 = np.random.randn(20, 2) * 0.5 + [0.2, 1.2]
    
    ax.scatter(c1[:,0], c1[:,1], color='#3B82F6', alpha=0.7, s=40, label='Vision Cluster A')
    ax.scatter(c2[:,0], c2[:,1], color='#10B981', alpha=0.7, s=40, label='Text Cluster B')
    ax.scatter(c3[:,0], c3[:,1], color='#8B5CF6', alpha=0.7, s=40, label='Audio Cluster C')
    
    # Query vector
    q = np.array([-1.0, 0.9])
    ax.scatter([q[0]], [q[1]], color='#EF4444', s=120, marker='*', zorder=5, label='Query Vector (q)')
    circle = plt.Circle(q, 0.65, color='#EF4444', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(q[0]+0.1, q[1]+0.2, "True Nearest\nNeighbors", fontsize=8.5, color='#B91C1C', fontweight='bold')
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.8, 2.2)
    ax.set_xlabel("Latent Dimension 1", fontsize=8.5, color='#64748B')
    ax.set_ylabel("Latent Dimension 2", fontsize=8.5, color='#64748B')
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.8)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 3: Linear Scan vs ANN Proximity Graph
    ax = axes[2]
    ax.set_title("3. Linear Scan vs. ANN Proximity Graph", fontsize=11, fontweight='bold', color='#1A2B4C', pad=12)
    ax.axis('off')
    
    card1 = dict(boxstyle='round,pad=0.5', facecolor='#FEE2E2', edgecolor='#EF4444', linewidth=1.2)
    card2 = dict(boxstyle='round,pad=0.5', facecolor='#ECFDF5', edgecolor='#10B981', linewidth=1.5)
    
    ax.text(0.5, 0.76, "BRUTE-FORCE LINEAR SCAN\n• Complexity: O(N · D)\n• 1,000,000 vectors = 128M FLOPs / query\n• Server latency: > 250 ms (Crashes at scale)", ha='center', va='center', bbox=card1, fontsize=8.5, color='#991B1B')
    
    ax.annotate("VS", xy=(0.5, 0.50), xytext=(0.5, 0.50), ha='center', va='center', fontsize=11, fontweight='bold', color='#64748B')
    
    ax.text(0.5, 0.24, "ANN PROXIMITY GRAPH SEARCH\n• Complexity: O(log N)\n• Evaluates only ~800 neighbor hops\n• 99.9% compute saved | Latency: < 1.5 ms\n• Accuracy: > 97.5% Recall@10", ha='center', va='center', bbox=card2, fontsize=8.5, color='#065F46', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram1_embeddings_ann.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved diagram1_embeddings_ann.png")


# =============================================================================
# DIAGRAM 2: HNSW Hierarchy & Layer Architecture (Standard vs Adaptive)
# =============================================================================
def make_diagram2():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    
    # Subplot 1: Standard HNSW Hierarchy
    ax = axes[0]
    ax.set_title("Standard HNSW: Multi-Layer Skip-List (Rigid M=16)", fontsize=11, fontweight='bold', color='#1A2B4C', pad=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # Layer 2
    ax.text(0.5, 7.5, "Layer 2 (Express Highway)", fontsize=9, fontweight='bold', color='#1E40AF')
    ax.plot([2.5, 7.5], [7.0, 7.0], 'o-', color='#3B82F6', lw=2.5, markersize=8)
    ax.text(5.0, 7.25, "Long-range skip links", ha='center', fontsize=7.5, color='#64748B')
    
    # Layer 1
    ax.text(0.5, 5.0, "Layer 1 (Transit Roads)", fontsize=9, fontweight='bold', color='#1E40AF')
    l1_x = [2.5, 4.2, 5.8, 7.5]
    ax.plot(l1_x, [4.5]*4, 's-', color='#60A5FA', lw=1.8, markersize=7)
    
    # Vertical inter-layer links
    ax.plot([2.5, 2.5], [7.0, 4.5], 'k:', lw=1.2, alpha=0.6)
    ax.plot([7.5, 7.5], [7.0, 4.5], 'k:', lw=1.2, alpha=0.6)
    
    # Layer 0
    ax.text(0.5, 2.4, "Layer 0 (Local Streets - All Nodes)", fontsize=9, fontweight='bold', color='#1E40AF')
    l0_x = np.linspace(1.5, 8.5, 11)
    ax.plot(l0_x, [1.8]*11, 'o-', color='#93C5FD', lw=1.2, markersize=6)
    for x in l1_x:
        ax.plot([x, x], [4.5, 1.8], 'k:', lw=1.2, alpha=0.6)
        
    # Flaw callout
    flaw_box = dict(boxstyle='round,pad=0.4', facecolor='#FEE2E2', edgecolor='#EF4444', lw=1)
    ax.text(5.0, 0.6, "FLAW IN STANDARD HNSW: Forces uniform M=16 everywhere.\nCentroids become congested super-hubs; sparse nodes starve.", ha='center', va='center', bbox=flaw_box, fontsize=8, color='#991B1B')
    
    # Subplot 2: AdaptiveVec Topology-Aware Architecture
    ax = axes[1]
    ax.set_title("AdaptiveVec: Topology-Aware Layer 0 + Hub Regulation", fontsize=11, fontweight='bold', color='#1A2B4C', pad=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # Layer 2 & 1 decoupled
    ax.text(0.5, 7.5, "Layer 2: Standard Uniform (Decoupled)", fontsize=9, fontweight='bold', color='#065F46')
    ax.plot([2.5, 7.5], [7.0, 7.0], 'o-', color='#10B981', lw=2.5, markersize=8)
    
    ax.text(0.5, 5.0, "Layer 1: Standard Uniform (Decoupled)", fontsize=9, fontweight='bold', color='#065F46')
    ax.plot(l1_x, [4.5]*4, 's-', color='#34D399', lw=1.8, markersize=7)
    
    ax.plot([2.5, 2.5], [7.0, 4.5], 'k:', lw=1.2, alpha=0.6)
    ax.plot([7.5, 7.5], [7.0, 4.5], 'k:', lw=1.2, alpha=0.6)
    for x in l1_x:
        ax.plot([x, x], [4.5, 1.8], 'k:', lw=1.2, alpha=0.6)
        
    # Layer 0 Adaptive
    ax.text(0.5, 2.4, "Layer 0: Dynamic M(x) Guided by Local Intrinsic Dimension (LID)", fontsize=9, fontweight='bold', color='#065F46')
    
    # Dense cluster (low M)
    ax.plot(l0_x[:4], [1.8]*4, 'o-', color='#059669', lw=1.0, markersize=5)
    ax.text(2.6, 1.3, "Dense Cluster\nM=10 (Pruned)", ha='center', fontsize=7.5, color='#047857', fontweight='bold')
    
    # Sparse ridge (high M)
    ax.plot(l0_x[4:], [1.8]*7, 'o-', color='#C97D4A', lw=2.2, markersize=7)
    ax.text(6.8, 1.3, "Sparse Boundary Ridge\nM=22 (Expanded)", ha='center', fontsize=7.5, color='#C97D4A', fontweight='bold')
    
    sol_box = dict(boxstyle='round,pad=0.4', facecolor='#ECFDF5', edgecolor='#10B981', lw=1)
    ax.text(5.0, 0.6, "ADAPTIVEVEC INNOVATION: Saves 7.4% edges & 28.1% build time.\nHub regulation (mu=0.15) slashes in-degree variance by 54.9%.", ha='center', va='center', bbox=sol_box, fontsize=8, color='#065F46', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram2_hnsw_hierarchy.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved diagram2_hnsw_hierarchy.png")


# =============================================================================
# DIAGRAM 3: Complete 5-Stage Algorithmic Flowchart
# =============================================================================
def make_diagram3():
    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    stages = [
        ("STAGE 1\nLID Probing", "O(N · k log k)\nOverhead < 0.8%\nLevina-Bickel Estimator", "#EFF6FF", "#2563EB", "#1E40AF"),
        ("STAGE 2\nDegree Scaling", "Dynamic M(x)\nLayer-Decoupled\n-28.1% Build Time", "#FEF3C7", "#D97706", "#92400E"),
        ("STAGE 3\nHubness Reg.", "mu = 0.15 Penalty\n-54.9% In-Degree Var\nEliminates Bottlenecks", "#F3E8FF", "#9333EA", "#6B21A8"),
        ("STAGE 4\nStagnation Exit", "p = 6, eps = 10^-4\n-30.1% Evals/Query\n+50.3% Search QPS!", "#ECFDF5", "#059669", "#047857"),
        ("STAGE 5\nAsymmetric SQ8", "8-Bit Affine Quant\nFP32 Top-k Rerank\n-62.4% RAM Footprint", "#FFF1F2", "#E11D48", "#9F1239"),
    ]
    
    x_positions = [1.5, 4.3, 7.1, 9.9, 12.7]
    
    for i, (title, desc, bg, edge, txt_col) in enumerate(stages):
        x = x_positions[i]
        box = dict(boxstyle='round,pad=0.5', facecolor=bg, edgecolor=edge, lw=1.5)
        ax.text(x, 2.7, title, ha='center', va='center', bbox=box, fontsize=9.5, fontweight='bold', color=txt_col)
        ax.text(x, 1.2, desc, ha='center', va='center', bbox=dict(boxstyle='square,pad=0.3', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.8), fontsize=8, color='#334155')
        
        # Connectors
        if i < len(stages) - 1:
            next_x = x_positions[i+1]
            ax.annotate("", xy=(next_x - 1.15, 2.7), xytext=(x + 1.15, 2.7),
                        arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))
            
    ax.text(7.5, 4.3, "ADAPTIVEVEC 5-STAGE END-TO-END SYSTEM PIPELINE", ha='center', fontsize=12, fontweight='bold', color='#1A2B4C')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram3_pipeline_flowchart.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved diagram3_pipeline_flowchart.png")


# =============================================================================
# DIAGRAM 4: Greedy Search Trajectory & Stagnation Exit
# =============================================================================
def make_diagram4():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=300)
    
    # Subplot 1: Spatial Graph Routing
    ax = axes[0]
    ax.set_title("A. Greedy Graph Traversal: Standard vs. Adaptive Exit", fontsize=10.5, fontweight='bold', color='#1A2B4C')
    
    np.random.seed(101)
    # Background nodes
    bg_x = np.random.uniform(1, 9, 45)
    bg_y = np.random.uniform(1, 9, 45)
    ax.scatter(bg_x, bg_y, color='#CBD5E1', s=25, alpha=0.6)
    
    # Trajectory points
    traj_x = [1.5, 2.8, 4.2, 5.5, 6.6, 7.3, 7.8, 8.1, 8.2, 8.25, 8.28, 8.30]
    traj_y = [1.5, 2.4, 3.8, 5.0, 6.2, 6.9, 7.3, 7.5, 7.6, 7.62, 7.63, 7.63]
    
    # Useful search hops
    ax.plot(traj_x[:7], traj_y[:7], 'o-', color='#3B82F6', lw=2, markersize=6, label='Active Routing (Hops 1-6)')
    # Stagnation window
    ax.plot(traj_x[6:], traj_y[6:], 's-', color='#10B981', lw=2.5, markersize=7, label='Stagnation Window (Hops 7-12, p=6)')
    
    # True nearest neighbor
    ax.scatter([8.30], [7.63], color='#EF4444', s=140, marker='*', zorder=10, label='True 1-NN Vector')
    
    # Early Exit trigger point
    ax.annotate("EARLY EXIT FIRED!\n(Hop 12: Delta_d < 10^-4)", xy=(8.30, 7.63), xytext=(5.6, 8.3),
                arrowprops=dict(facecolor='#10B981', edgecolor='#047857', arrowstyle="->", lw=1.8),
                fontsize=8.5, fontweight='bold', color='#047857',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECFDF5', edgecolor='#10B981'))
    
    # Wasted standard HNSW hops
    wasted_x = [8.31, 8.29, 8.32, 8.30]
    wasted_y = [7.64, 7.62, 7.63, 7.64]
    ax.plot(wasted_x, wasted_y, 'x--', color='#EF4444', lw=1.2, markersize=6, label='Wasted Standard HNSW Hops')
    
    ax.set_xlabel("Vector Dimension X", fontsize=8.5)
    ax.set_ylabel("Vector Dimension Y", fontsize=8.5)
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 2: Distance vs Hop Stagnation Curve
    ax = axes[1]
    ax.set_title("B. Distance Improvement vs. Search Hop Depth", fontsize=10.5, fontweight='bold', color='#1A2B4C')
    
    hops = np.arange(1, 21)
    dist = 0.85 * np.exp(-0.35 * hops) + 0.0879
    # Add tiny jitter to simulate flatline
    dist[11:] = 0.0879 + np.random.uniform(0, 0.00003, len(dist[11:]))
    
    ax.plot(hops[:12], dist[:12], 'o-', color='#3B82F6', lw=2, label='Rapid Centroid Convergence')
    ax.plot(hops[11:], dist[11:], 's-', color='#EF4444', lw=1.5, label='Stagnation Plateau (Wasted Compute)')
    
    # Highlight Early Exit line
    ax.axvline(12, color='#10B981', linestyle='--', lw=2, label='AdaptiveVec Exit (p=6, Hop 12)')
    
    ax.text(12.3, 0.45, "SAVES 30.1% EVALUATIONS!\n(Throughput +50.3% QPS)", fontsize=8.5, fontweight='bold', color='#047857',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECFDF5', edgecolor='#10B981'))
    
    ax.set_xlabel("Greedy Search Hop Number", fontsize=8.5)
    ax.set_ylabel("Euclidean Distance to Query (dist)", fontsize=8.5)
    ax.set_ylim(0.0, 0.9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram4_stagnation_trajectory.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved diagram4_stagnation_trajectory.png")


# =============================================================================
# DIAGRAM 5: Hubness Regulation Visual Comparison
# =============================================================================
def make_diagram5():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=300)
    
    # Subplot 1: Unregulated Graph (Severe Hubness)
    ax = axes[0]
    ax.set_title("Unregulated HNSW (mu = 0.0): Central Super-Hub", fontsize=10.5, fontweight='bold', color='#991B1B')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')
    
    # Central Hub
    ax.scatter([0], [0], color='#EF4444', s=350, zorder=5)
    ax.text(0, -0.4, "SUPER-HUB\n(In-Degree = 18)\nVariance = 142.8", ha='center', fontsize=8.5, fontweight='bold', color='#991B1B')
    
    # Surrounding nodes pointing to central hub
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for ang in angles:
        px = 1.8 * np.cos(ang)
        py = 1.8 * np.sin(ang)
        ax.scatter([px], [py], color='#3B82F6', s=50)
        ax.annotate("", xy=(0.15*np.cos(ang), 0.15*np.sin(ang)), xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#EF4444', alpha=0.7))
        
    ax.text(0, 2.2, "TRAFFIC BOTTLENECK: Queries get trapped in the hub", ha='center', fontsize=8, color='#991B1B', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor='#EF4444'))
    
    # Subplot 2: Regulated Graph (mu = 0.15)
    ax = axes[1]
    ax.set_title("AdaptiveVec Regulated (mu = 0.15): Balanced Topology", fontsize=10.5, fontweight='bold', color='#065F46')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')
    
    # Multiple distributed transit nodes
    transit_x = [-0.6, 0.6, 0.0]
    transit_y = [-0.3, -0.3, 0.7]
    ax.scatter(transit_x, transit_y, color='#10B981', s=160, zorder=5)
    ax.text(0, -1.1, "BALANCED TRANSIT NODES\nIn-Degree Variance = 64.4 (-54.9%)", ha='center', fontsize=8.5, fontweight='bold', color='#065F46')
    
    for i, ang in enumerate(angles):
        px = 1.8 * np.cos(ang)
        py = 1.8 * np.sin(ang)
        ax.scatter([px], [py], color='#3B82F6', s=50)
        target = i % 3
        tx = transit_x[target]
        ty = transit_y[target]
        ax.annotate("", xy=(tx, ty), xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#10B981', alpha=0.7))
        
    ax.text(0, 2.2, "UNIFORM TRAFFIC: Multi-route distributed highway", ha='center', fontsize=8, color='#065F46', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECFDF5', edgecolor='#10B981'))
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram5_hubness_comparison.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved diagram5_hubness_comparison.png")


# =============================================================================
# DIAGRAM 6: Asymmetric SQ8 Quantization Memory Architecture
# =============================================================================
def make_diagram6():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), dpi=300)
    
    # Subplot 1: Memory Layout Comparison
    ax = axes[0]
    ax.set_title("Memory Layout: FP32 vs. 8-Bit Asymmetric SQ8", fontsize=10.5, fontweight='bold', color='#1A2B4C')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # FP32 Box
    ax.text(0.5, 4.8, "Standard HNSW Vector: 32-Bit Float (FP32)", fontsize=9, fontweight='bold', color='#1E40AF')
    ax.text(0.5, 4.0, "128 dims * 4 bytes/dim = 512 Bytes / Vector (59.9 MB on SIFT-100K)", fontsize=8, color='#64748B')
    fp_colors = ['#93C5FD', '#BFDBFE', '#DBEAFE', '#EFF6FF']
    for d in range(4):
        ax.text(1.2 + d*2.1, 3.2, f"Dim {d}: 4 Bytes\n[Sign | Exp | Mantissa]", ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.4', facecolor=fp_colors[d], edgecolor='#3B82F6', lw=1), fontsize=7.5)
        
    # SQ8 Box
    ax.text(0.5, 2.0, "AdaptiveVec Regime B: 8-Bit Quantized (SQ8)", fontsize=9, fontweight='bold', color='#065F46')
    ax.text(0.5, 1.2, "128 dims * 1 byte/dim = 128 Bytes / Vector (22.5 MB on SIFT-100K: -62.4% RAM)", fontsize=8, color='#047857', fontweight='bold')
    for d in range(4):
        ax.text(1.2 + d*2.1, 0.4, f"Dim {d}: 1 Byte\n[uint8: 0..255]", ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.4', facecolor='#A7F3D0', edgecolor='#10B981', lw=1.2), fontsize=7.5, fontweight='bold')
        
    # Subplot 2: 2-Stage Re-Ranking Pipeline
    ax = axes[1]
    ax.set_title("2-Stage Search: INT8 Beam Search + FP32 Top-k Rerank", fontsize=10.5, fontweight='bold', color='#1A2B4C')
    ax.axis('off')
    
    b1 = dict(boxstyle='round,pad=0.5', facecolor='#FEF3C7', edgecolor='#D97706', lw=1.2)
    b2 = dict(boxstyle='round,pad=0.5', facecolor='#ECFDF5', edgecolor='#10B981', lw=1.5)
    
    ax.text(0.5, 0.72, "STAGE 1: FAST INT8 GRAPH ROUTING\n• Uses _mm256_maddubs_epi16 SIMD integer dot-products\n• 32 components computed per cycle | Consumes 62.4% less memory\n• Navigates beam search down to Top-50 candidate pool", ha='center', va='center', bbox=b1, fontsize=8, color='#92400E')
    
    ax.annotate("", xy=(0.5, 0.44), xytext=(0.5, 0.54), arrowprops=dict(arrowstyle="->", lw=1.8, color='#059669'))
    
    ax.text(0.5, 0.22, "STAGE 2: FP32 EXACT RE-RANKING\n• Reads true 32-bit floats strictly for top-50 candidates\n• Restores exact ranking precision\n• Delivers 95.9% Recall@10 with massive memory savings!", ha='center', va='center', bbox=b2, fontsize=8, color='#065F46', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram6_sq8_quantization.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved diagram6_sq8_quantization.png")

if __name__ == "__main__":
    make_diagram1()
    make_diagram2()
    make_diagram3()
    make_diagram4()
    make_diagram5()
    make_diagram6()
    print("All 6 handbook diagrams generated successfully.")
