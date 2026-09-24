import React from 'react';
import { BookOpen, Cpu, Copy, Check } from 'lucide-react';
import { MathView } from '../components/MathView';

export const TheoryTab: React.FC = () => {
  const [copiedBib, setCopiedBib] = React.useState(false);

  const bibtex = `@article{adaptivevec2026,
  title   = {AdaptiveVec: Density- and Dimension-Aware Proximity Graph Index for High-Dimensional Vector Search},
  journal = {Proceedings of Engineering Design & Innovation (EDI)},
  volume  = {14},
  number  = {1},
  pages   = {1--13},
  year    = {2026}
}`;

  const copyBibtex = () => {
    navigator.clipboard.writeText(bibtex);
    setCopiedBib(true);
    setTimeout(() => setCopiedBib(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span className="badge-pill accent">Mathematical Formulation</span>
          <span className="badge-pill">IEEE / ACM Standard</span>
        </div>
        <h2 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>
          Theoretical Foundations & Engine Microarchitecture
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
          Mathematical observations, topological bounds, and hardware SIMD acceleration specifications powering AdaptiveVec.
        </p>
      </div>

      {/* Grid of 2 Formal Mathematical Callouts */}
      <div className="grid-2">
        {/* Callout 1: Observation 1 (LID Estimator) */}
        <div className="instrument-card" style={{ borderLeft: '3px solid var(--accent)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <span className="badge-pill accent">Empirical Observation 1</span>
            <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Local Intrinsic Dimensionality (LID) Bounds
            </span>
          </div>

          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
            Let <em>P</em> be a dataset sampled from a Riemannian manifold embedded in <strong>R</strong><sup>D</sup>. The local intrinsic dimensionality around node <em>x</em> is estimated via extreme value theory on nearest-neighbor distance ratios:
          </p>

          <MathView
            title="Levina-Bickel Extreme Value Estimator"
            latex="\\widehat{\\text{LID}}(x) = -\\left[ \\frac{1}{k} \\sum_{i=1}^{k} \\ln \\left( \\frac{r_i(x)}{r_k(x)} \\right) \\right]^{-1}"
            fallbackText="LIDest(x) = - [ (1 / k) ∑_{i=1}^k ln( r_i(x) / r_k(x) ) ]⁻¹"
            variables={[
              { symbol: 'k = 16', name: 'Neighborhood Size', desc: 'Sample window for extreme value estimation' },
              { symbol: 'r_i(x)', name: 'Sample Radius', desc: 'Euclidean distance to the i-th nearest neighbor' },
              { symbol: 'r_k(x)', name: 'Boundary Horizon', desc: 'Distance to the k-th neighbor in local manifold window' },
            ]}
            significance="Scales in O(N · k log k) complexity, consuming under 0.8% of index construction time while separating real manifold geometry from ambient noise."
            accentColor="var(--accent-glow)"
          />
        </div>

        {/* Callout 2: Proposition 1 (Stagnation Search Convergence) */}
        <div className="instrument-card" style={{ borderLeft: '3px solid var(--semantic-emerald)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <span className="badge-pill emerald">Proposition 1</span>
            <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Search Convergence Invariance under Stagnation
            </span>
          </div>

          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
            In greedy graph beam search over a connected Delaunay-approximation graph, let <em>d<sub>t</sub></em> denote the distance from query point <em>q</em> to the best candidate at hop <em>t</em>. If:
          </p>

          <MathView
            title="Distance Stagnation Termination Condition"
            latex="\\sum_{j=0}^{p-1} \\big| d_{t-j} - d_{t-j-1} \\big| < p \\cdot \\varepsilon \\implies \\mathbf{CONVERGENCE\\_EXIT}"
            fallbackText="∑_{j=0}^{p-1} | d_{t-j} - d_{t-j-1} | < p · ε   (with p = 6, ε = 10⁻⁴)"
            variables={[
              { symbol: 'p = 6', name: 'Patience Steps', desc: 'Number of consecutive search hops evaluated without distance progress' },
              { symbol: 'ε = 10⁻⁴', name: 'Convergence Tolerance', desc: 'Minimum meaningful distance improvement threshold' },
              { symbol: 'd_t', name: 'Current Minimum', desc: 'Euclidean distance of best-so-far candidate to query q' },
            ]}
            significance="The probability of discovering an unexplored candidate satisfying dist(q, v) < d_t - δ decays exponentially. Exiting early eliminates 30.1% redundant evals with negligible recall loss (-1.68%)."
            accentColor="var(--semantic-emerald)"
          />
        </div>
      </div>

      {/* C++ AVX2 Hardware Vector Engine Architecture */}
      <div className="instrument-card" style={{ padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Cpu size={16} color="var(--accent-glow)" />
          <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
            C++20 AVX2 / FMA Vector SIMD Microarchitecture
          </h3>
          <span className="badge-pill emerald">x86_64 Native</span>
        </div>

        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.55, marginBottom: 12 }}>
          To achieve production-grade search throughput, the core distance computation routines are written in native C++ using AVX2 and FMA (Fused Multiply-Add) vector intrinsics, processing 8 single-precision floats per CPU clock cycle:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 16 }}>
          <div className="mono" style={{
            background: 'var(--bg-surface-sunken)',
            border: '1px solid var(--border-default)',
            borderRadius: 'var(--radius-xs)',
            padding: 14,
            fontSize: '11px',
            color: 'var(--text-primary)',
            overflowX: 'auto',
            lineHeight: 1.6,
          }}>
            <span style={{ color: 'var(--text-dim)' }}>// AVX2 8-Wide FMA Euclidean Distance Loop</span>{'\n'}
            __m256 sum = _mm256_setzero_ps();{'\n'}
            <span style={{ color: 'var(--semantic-amber)' }}>for</span> (size_t d = 0; d &lt; dim; d += 8) {'{'}{'\n'}
            {'  '}__m256 va = _mm256_loadu_ps(a + d);{'\n'}
            {'  '}__m256 vb = _mm256_loadu_ps(b + d);{'\n'}
            {'  '}__m256 diff = _mm256_sub_ps(va, vb);{'\n'}
            {'  '}sum = <span style={{ color: 'var(--accent-glow)', fontWeight: 'bold' }}>_mm256_fmadd_ps</span>(diff, diff, sum);{'\n'}
            {'}'}{'\n'}
            <span style={{ color: 'var(--semantic-emerald)' }}>return</span> _mm256_reduce_add_ps(sum);
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, justifyContent: 'center' }}>
            <div style={{ background: 'var(--bg-card-subtle)', padding: 10, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>Memory Alignment</div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                64-Byte Cache Line Aligned Vectors
              </div>
              <div style={{ fontSize: '10.5px', color: 'var(--text-dim)', marginTop: 2 }}>
                Prevents false sharing and unaligned memory access penalties.
              </div>
            </div>

            <div style={{ background: 'var(--bg-card-subtle)', padding: 10, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>Quantization Acceleration</div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--semantic-emerald)', marginTop: 2 }}>
                _mm256_maddubs_epi16 Asymmetric Dot-Product
              </div>
              <div style={{ fontSize: '10.5px', color: 'var(--text-dim)', marginTop: 2 }}>
                Processes 32 INT8 components per instruction in Regime B.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* BibTeX Citation Box */}
      <div className="instrument-card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BookOpen size={14} color="var(--accent-glow)" />
            <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
              BibTeX Citation Format
            </span>
          </div>

          <button
            onClick={copyBibtex}
            className="btn-instrument"
            style={{ fontSize: '11px', padding: '4px 10px' }}
          >
            {copiedBib ? <Check size={12} color="var(--semantic-emerald)" /> : <Copy size={12} />}
            <span>{copiedBib ? 'Copied!' : 'Copy BibTeX'}</span>
          </button>
        </div>

        <pre className="mono" style={{
          background: 'var(--bg-surface-sunken)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-xs)',
          padding: 12,
          fontSize: '11px',
          color: 'var(--text-secondary)',
          overflowX: 'auto',
          lineHeight: 1.5,
        }}>
          {bibtex}
        </pre>
      </div>
    </div>
  );
};
