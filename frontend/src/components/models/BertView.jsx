import React, { useState } from 'react';
import { Smile, Frown, Meh, Sparkles, BarChart2, Star } from 'lucide-react';

const PRESETS = [
  "Support resolved my ticket within minutes, absolutely brilliant and helpful service!",
  "The app crashes every time I try to export an invoice, completely unacceptable downtime.",
  "The policy meeting is scheduled for Tuesday at 3 PM in conference room B.",
  "The new dashboard is intuitive and saves our engineering team hours every week.",
  "Worst billing update ever, half the account features are broken and unhelpful."
];

export default function BertView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [text, setText] = useState(PRESETS[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleClassify = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/bert/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, explain: true })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const getLabelColor = (label) => {
    if (label === 'positive') return 'var(--accent-emerald)';
    if (label === 'negative') return 'var(--accent-rose)';
    return 'var(--accent-amber)';
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '24px' }}>
      {/* Left Column: Input & Presets */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>BERT Sentiment Classification</h2>
            <span className="glass-pill" style={{ color: 'var(--accent-emerald)' }}>
              167M Params · Bi-directional Encoder
            </span>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '6px' }}>
            Multi-class feedback classifier powered by BERT. Includes word-level occlusion sensitivity explanations to reveal token-level sentiment impact.
          </p>
        </div>

        {/* Sample Feedbacks */}
        <div>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            SAMPLE FEEDBACK STATEMENTS
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {PRESETS.map((p, i) => (
              <button
                key={i}
                className="glass-card"
                onClick={() => setText(p)}
                style={{
                  padding: '8px 12px',
                  fontSize: '0.78rem',
                  textAlign: 'left',
                  cursor: 'pointer',
                  border: text === p ? '1px solid var(--accent-emerald)' : '1px solid var(--glass-border-subtle)',
                  background: text === p ? 'var(--glass-bg-hover)' : 'var(--glass-subtle)'
                }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {/* Text Input */}
        <div>
          <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            CUSTOMER / EMPLOYEE TEXT
          </label>
          <textarea
            className="glass-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder="Type customer or employee feedback here..."
          />
        </div>

        <button
          className="glass-btn glass-btn-primary"
          onClick={handleClassify}
          disabled={loading || !text.trim()}
          style={{ padding: '12px', marginTop: 'auto' }}
        >
          <Sparkles size={16} className={loading ? 'spin' : ''} />
          <span>{loading ? 'Analyzing Sentiment & Tokens...' : 'Classify Sentiment'}</span>
        </button>
      </div>

      {/* Right Column: Probabilities & Token Occlusion Explanation */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Model Prediction & Diagnostics</h3>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('bert')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run F1 Benchmark'}
          </button>
        </div>

        {result && (
          <>
            {/* Classification Card */}
            <div className="glass-card" style={{ padding: '18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <span className="metric-lbl">PREDICTED SENTIMENT</span>
                <div style={{ 
                  fontSize: '1.4rem', 
                  fontWeight: 800, 
                  color: getLabelColor(result.label),
                  textTransform: 'uppercase',
                  marginTop: '4px'
                }}>
                  {result.label}
                </div>
                {result.stars && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
                    {[1, 2, 3, 4, 5].map((s) => (
                      <Star 
                        key={s} 
                        size={14} 
                        fill={s <= Math.round(result.stars) ? 'var(--accent-amber)' : 'none'} 
                        color="var(--accent-amber)" 
                      />
                    ))}
                    <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', marginLeft: '4px' }}>
                      {result.stars} / 5.0
                    </span>
                  </div>
                )}
              </div>

              <div style={{ textAlign: 'right' }}>
                <span className="metric-lbl">CONFIDENCE</span>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, fontFamily: 'var(--font-mono)' }}>
                  {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Probability Bars */}
            {result.probs && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(result.probs).map(([lbl, p]) => (
                  <div key={lbl} style={{ fontSize: '0.78rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                      <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{lbl}</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '8px', background: 'var(--glass-subtle)', borderRadius: '999px', overflow: 'hidden' }}>
                      <div style={{ 
                        width: `${p * 100}%`, 
                        height: '100%', 
                        background: getLabelColor(lbl),
                        borderRadius: '999px'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Word-Level Occlusion Explanation */}
            {result.tokens && (
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
                  WORD OCCLUSION IMPACT (HEATMAP)
                </div>
                <div className="glass-card" style={{ 
                  padding: '14px', 
                  display: 'flex', 
                  flexWrap: 'wrap', 
                  gap: '6px',
                  background: 'var(--glass-subtle)',
                  lineHeight: '1.8'
                }}>
                  {result.tokens.map((t, i) => {
                    const impact = t.impact;
                    const isPos = impact > 0.02;
                    const isNeg = impact < -0.02;
                    const bg = isPos 
                      ? `rgba(52, 211, 153, ${Math.min(0.6, impact * 1.5)})`
                      : isNeg 
                        ? `rgba(251, 113, 133, ${Math.min(0.6, Math.abs(impact) * 1.5)})`
                        : 'transparent';

                    return (
                      <span 
                        key={i} 
                        title={`Impact: ${impact > 0 ? '+' : ''}${impact}`}
                        style={{ 
                          padding: '2px 6px', 
                          borderRadius: '4px', 
                          background: bg,
                          border: (isPos || isNeg) ? '1px solid var(--glass-border-subtle)' : 'none',
                          fontSize: '0.85rem'
                        }}
                      >
                        {t.word}
                      </span>
                    );
                  })}
                </div>
                <div style={{ display: 'flex', gap: '16px', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                  <span style={{ color: 'var(--accent-emerald)' }}>■ Positive polarity driver</span>
                  <span style={{ color: 'var(--accent-rose)' }}>■ Negative polarity driver</span>
                </div>
              </div>
            )}
          </>
        )}

        {!result && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', fontStyle: 'italic', textAlign: 'center', padding: '40px' }}>
            Click "Classify Sentiment" to inspect confidence scores and word-level occlusion sensitivity.
          </div>
        )}
      </div>
    </div>
  );
}
