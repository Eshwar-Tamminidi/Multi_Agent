import React, { useState } from 'react';
import { Play, Sparkles, Sliders, RefreshCw, Cpu, BookOpen } from 'lucide-react';

const PRESETS = [
  { label: 'Shakespeare Soliloquy', prompt: 'ROMEO:\n' },
  { label: 'Enterprise Legal Clause', prompt: 'SECTION 4. Confidentiality and Proprietary Information:\n' },
  { label: 'Technical Specification', prompt: 'ARCHITECTURE:\nThe multi-modal neural network operates by ' },
  { label: 'Customer Inquiry', prompt: 'Customer: How can I change my corporate leave schedule?\nSupport: ' }
];

export default function CharGptView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [prompt, setPrompt] = useState('ROMEO:\n');
  const [maxTokens, setMaxTokens] = useState(300);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/char_gpt/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_new_tokens: parseInt(maxTokens, 10),
          temperature: parseFloat(temperature),
          top_k: parseInt(topK, 10)
        })
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

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
      {/* Left Column: Controls & Prompt */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>Character-Level GPT</h2>
            <span className="glass-pill" style={{ color: 'var(--accent-cyan)' }}>
              ~1.8M Params · Trained from scratch
            </span>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '6px' }}>
            Predicts the next character autoregressively using a lightweight decoder-only Transformer. Demonstrates how neural representations emerge from raw character sequences.
          </p>
        </div>

        {/* Prompt presets */}
        <div>
          <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            SAMPLE PROMPTS
          </label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {PRESETS.map((p, i) => (
              <button
                key={i}
                className="glass-btn"
                onClick={() => setPrompt(p.prompt)}
                style={{ padding: '6px 12px', fontSize: '0.75rem' }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Prompt Input */}
        <div>
          <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            INPUT PROMPT
          </label>
          <textarea
            className="glass-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Type your starting prompt here..."
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}
          />
        </div>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Tokens</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{maxTokens}</span>
            </div>
            <input
              type="range"
              min="50"
              max="1000"
              step="50"
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Temperature</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{temperature}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Top-K</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{topK}</span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
            />
          </div>
        </div>

        <button
          className="glass-btn glass-btn-primary"
          onClick={handleGenerate}
          disabled={loading}
          style={{ padding: '12px', width: '100%', marginTop: 'auto' }}
        >
          <Sparkles size={16} className={loading ? 'spin' : ''} />
          <span>{loading ? 'Synthesizing Text...' : 'Generate Completion'}</span>
        </button>
      </div>

      {/* Right Column: Output & Diagnostics */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Autoregressive Output</h3>
          {result?.metrics && (
            <div style={{ display: 'flex', gap: '8px' }}>
              <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>
                ⚡ {result.metrics.inference_ms} ms
              </span>
              <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>
                💾 {result.metrics.rss_mb} MB RSS
              </span>
            </div>
          )}
        </div>

        {error && (
          <div className="glass-card" style={{ padding: '12px', borderLeft: '4px solid var(--accent-rose)', color: 'var(--accent-rose)', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}

        <div className="glass-card" style={{ 
          flex: 1, 
          padding: '16px', 
          fontFamily: 'var(--font-mono)', 
          fontSize: '0.88rem',
          lineHeight: '1.6',
          whiteSpace: 'pre-wrap',
          overflowY: 'auto',
          maxHeight: '420px',
          background: 'var(--glass-subtle)'
        }}>
          {result ? (
            <>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{result.prompt}</span>
              <span style={{ color: 'var(--text-primary)' }}>{result.completion}</span>
            </>
          ) : (
            <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Generated character stream will appear here after clicking "Generate Completion"...
            </span>
          )}
        </div>

        {/* Model info & Training Curve indicator */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>Vocabulary: 65 characters · 4 Layers · 4 Heads · 192 Dim</span>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('char_gpt')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Evaluating...' : 'Run Perplexity Benchmark'}
          </button>
        </div>
      </div>
    </div>
  );
}
