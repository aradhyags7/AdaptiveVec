import React, { useEffect, useState } from 'react';

interface MathVariable {
  symbol: string;
  name: string;
  desc: string;
}

interface MathViewProps {
  title?: string;
  tag?: string;
  latex: string;
  fallbackText: string;
  variables?: MathVariable[];
  significance?: string;
  accentColor?: string;
}

export const MathView: React.FC<MathViewProps> = ({
  title = 'Mathematical Formulation',
  tag,
  latex,
  fallbackText,
  variables = [],
  significance,
  accentColor = 'var(--accent-glow)',
}) => {
  const [renderedHtml, setRenderedHtml] = useState<string | null>(null);

  useEffect(() => {
    const renderKatex = () => {
      const katex = (window as any).katex;
      if (katex) {
        try {
          const html = katex.renderToString(latex, {
            displayMode: true,
            throwOnError: false,
            strict: false,
          });
          setRenderedHtml(html);
          return true;
        } catch (err) {
          console.warn('KaTeX render error:', err);
        }
      }
      return false;
    };

    // Try rendering immediately
    if (!renderKatex()) {
      // If KaTeX script is still loading via CDN, check after brief intervals
      const interval = setInterval(() => {
        if (renderKatex()) {
          clearInterval(interval);
        }
      }, 200);

      const timeout = setTimeout(() => clearInterval(interval), 3000);
      return () => {
        clearInterval(interval);
        clearTimeout(timeout);
      };
    }
  }, [latex]);

  return (
    <div className="math-formula-container" style={{ borderLeft: `3px solid ${accentColor}` }}>
      {/* Header Bar */}
      <div className="math-formula-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: accentColor }}>{title}</span>
          {tag && <span className="badge-pill illustrative">{tag}</span>}
        </div>
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          CLOSED-FORM DERIVATION
        </span>
      </div>

      {/* Main Equation Box */}
      <div className="math-display-equation">
        {renderedHtml ? (
          <div
            dangerouslySetInnerHTML={{ __html: renderedHtml }}
            style={{ display: 'flex', justifyContent: 'center' }}
          />
        ) : (
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '15px',
            color: '#FFF9F0',
            letterSpacing: '0.01em',
            padding: '4px 0',
          }}>
            {fallbackText}
          </div>
        )}
      </div>

      {/* Variable Descriptions Legend */}
      {variables.length > 0 && (
        <div className="math-variable-grid">
          {variables.map((v, i) => (
            <div key={i} className="math-variable-item">
              <span className="math-var-symbol">{v.symbol}</span>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>{v.name}:</strong>{' '}
                <span style={{ color: 'var(--text-muted)' }}>{v.desc}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Significance Callout */}
      {significance && (
        <div style={{
          marginTop: 12,
          padding: '8px 12px',
          borderRadius: 'var(--radius-xs)',
          background: 'rgba(232, 162, 92, 0.08)',
          border: '1px solid rgba(232, 162, 92, 0.2)',
          fontSize: '11.5px',
          lineHeight: 1.5,
          color: 'var(--text-secondary)',
        }}>
          <strong style={{ color: accentColor }}>Theoretical Impact:</strong> {significance}
        </div>
      )}
    </div>
  );
};
