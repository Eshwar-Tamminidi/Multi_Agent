import React, { useState } from 'react';
import { Search, Sparkles, Zap, Layers, CheckCircle2, ArrowRight } from 'lucide-react';

const SAMPLE_QUERIES = [
  "password expired how to change",
  "book holidays next month",
  "pay me back for taxi fare",
  "my notebook is broken need another",
  "someone asked for my credentials by email",
  "the office is too hot"
];

export default function TwoTowerView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (q = query) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/two_tower/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, k: 4 })
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Search Header Panel */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>
                Two-Tower Semantic Search
              </h2>
              <span className="glass-pill" style={{ color: 'var(--accent-blue)' }}>
                Dual-Encoder with Contrastive InfoNCE Fine-Tuning
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0' }}>
              Maps queries and enterprise catalog services into a shared embedding space. Sub-millisecond ANN search compared against frozen embeddings and TF-IDF.
            </p>
          </div>

          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('two_tower')}
            disabled={isBenchmarking}
            style={{ fontSize: '0.82rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run Recall@3 Benchmark'}
          </button>
        </div>

        {/* Query Input */}
        <div style={{ display: 'flex', gap: '10px', marginTop: '16px' }}>
          <input
            type="text"
            className="glass-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type user intent (e.g., 'forgot password', 'vacation approval', 'order monitor')..."
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button
            className="glass-btn glass-btn-primary"
            onClick={() => handleSearch()}
            disabled={loading}
            style={{ padding: '0 24px', whiteSpace: 'nowrap' }}
          >
            <Search size={16} className={loading ? 'spin' : ''} />
            <span>{loading ? 'Searching...' : 'Search Catalog'}</span>
          </button>
        </div>

        {/* Query Presets */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '12px' }}>
          {SAMPLE_QUERIES.map((sq, i) => (
            <button
              key={i}
              className="glass-btn"
              onClick={() => {
                setQuery(sq);
                handleSearch(sq);
              }}
              style={{ padding: '4px 10px', fontSize: '0.72rem' }}
            >
              {sq}
            </button>
          ))}
        </div>
      </div>

      {/* 3-Column Comparative Retrieval View */}
      {result?.methods && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
          {/* Method 1: Contrastive Two-Tower */}
          <div className="glass-panel" style={{ padding: '20px', border: '1px solid var(--accent-blue)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <span className="glass-pill" style={{ background: 'var(--accent-glow)', color: 'var(--accent-blue)', fontWeight: 700 }}>
                  ★ Fine-Tuned Two-Tower
                </span>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Learned Query & Item Projections
                </div>
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600 }}>
                ⚡ {result.methods.two_tower.ms} ms
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {result.methods.two_tower.results.map((item, idx) => (
                <div key={idx} className="glass-card" style={{ padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>
                      #{idx + 1} {item.title}
                    </span>
                    <span className="glass-pill" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '4px 0 0', lineHeight: '1.4' }}>
                    {item.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Method 2: Frozen Sentence-BERT */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <span className="glass-pill" style={{ color: 'var(--accent-purple)' }}>
                  Frozen S-BERT Baseline
                </span>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Off-the-shelf MiniLM Embeddings
                </div>
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600 }}>
                ⚡ {result.methods.frozen_sbert.ms} ms
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {result.methods.frozen_sbert.results.map((item, idx) => (
                <div key={idx} className="glass-card" style={{ padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>
                      #{idx + 1} {item.title}
                    </span>
                    <span className="glass-pill" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '4px 0 0', lineHeight: '1.4' }}>
                    {item.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Method 3: Classical TF-IDF */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <span className="glass-pill" style={{ color: 'var(--text-muted)' }}>
                  TF-IDF Keyword Baseline
                </span>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Lexical Token Matching
                </div>
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 600 }}>
                ⚡ {result.methods.tfidf.ms} ms
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {result.methods.tfidf.results.map((item, idx) => (
                <div key={idx} className="glass-card" style={{ padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>
                      #{idx + 1} {item.title}
                    </span>
                    <span className="glass-pill" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '4px 0 0', lineHeight: '1.4' }}>
                    {item.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
