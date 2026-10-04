import React, { useState, useEffect } from 'react';
import { Database, Search, Upload, FileText, CheckCircle2, Sparkles, BookOpen, Layers } from 'lucide-react';

const SAMPLE_QUERIES = [
  "How many days of annual leave do full-time employees get?",
  "What is the minimum corporate password length and rotation policy?",
  "How much is the one-time home office stipend?",
  "What is the hotel cap in high-cost cities like New York and London?",
  "How fast must an incident commander be assigned for SEV1 incidents?"
];

export default function RagView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [useLlm, setUseLlm] = useState(false);
  const [k, setK] = useState(3);
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/rag/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleAsk = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/rag/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k: parseInt(k, 10), use_llm: useLlm })
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/rag/upload', { method: 'POST', body: fd });
      if (res.ok) {
        const data = await res.json();
        setUploadStatus(`Indexed ${data.added_chunks} new chunks into vector database!`);
        fetchDocuments();
      }
    } catch (e) {
      setUploadStatus('Upload failed: ' + e.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
      {/* Left Column: Query & Response */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>
              RAG with Sentence-BERT Embeddings
            </h2>
            <span className="glass-pill" style={{ color: 'var(--accent-indigo)' }}>
              all-MiniLM-L6-v2 · 384 Dim Vector Index
            </span>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '6px' }}>
            Dual-stage enterprise retrieval: Chunks Markdown knowledge documents into dense semantic embeddings. Cosine search selects top passages, followed by either neural sentence extraction or LLM synthesis.
          </p>
        </div>

        {/* Query Presets */}
        <div>
          <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            ENTERPRISE TEST QUESTIONS
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {SAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                className="glass-card"
                onClick={() => setQuery(q)}
                style={{ 
                  padding: '8px 12px', 
                  fontSize: '0.78rem', 
                  textAlign: 'left',
                  cursor: 'pointer',
                  border: query === q ? '1px solid var(--accent-cyan)' : '1px solid var(--glass-border-subtle)',
                  background: query === q ? 'var(--glass-bg-hover)' : 'var(--glass-subtle)'
                }}
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Query Input */}
        <div>
          <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            NATURAL LANGUAGE QUERY
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              className="glass-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything about enterprise policies..."
              onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
            />
            <button
              className="glass-btn glass-btn-primary"
              onClick={handleAsk}
              disabled={loading}
              style={{ padding: '0 20px', whiteSpace: 'nowrap' }}
            >
              <Search size={16} className={loading ? 'spin' : ''} />
              <span>{loading ? 'Searching...' : 'Retrieve'}</span>
            </button>
          </div>
        </div>

        {/* Options */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--glass-subtle)', borderRadius: 'var(--radius-sm)' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontSize: '0.85rem' }}>
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              style={{ width: '16px', height: '16px', accentColor: 'var(--accent-purple)' }}
            />
            <span>Synthesize Answer using Qwen 2.5 LLM (Generative RAG)</span>
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem' }}>
            <span>Top-K:</span>
            <select 
              value={k} 
              onChange={(e) => setK(e.target.value)}
              className="glass-select"
              style={{ padding: '4px 8px', width: 'auto' }}
            >
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="5">5</option>
            </select>
          </div>
        </div>

        {/* Answer Box */}
        {result && (
          <div className="glass-card" style={{ padding: '18px', background: 'var(--glass-bg-active)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span className="glass-pill" style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>
                Answer · {result.mode}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                Retrieval: {result.retrieval_ms} ms {result.generation_ms ? `· Gen: ${result.generation_ms} ms` : ''}
              </span>
            </div>
            <div style={{ fontSize: '0.95rem', lineHeight: '1.6', fontWeight: 500 }}>
              {result.answer || 'No direct answer could be extracted from top matches.'}
            </div>
          </div>
        )}
      </div>

      {/* Right Column: Retrieved Chunks & Documents */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Retrieved Evidence Chunks</h3>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('rag')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run RAG Benchmark'}
          </button>
        </div>

        {/* Chunks List */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '420px' }}>
          {result?.sources?.map((chunk, i) => (
            <div key={i} className="glass-card" style={{ padding: '14px', borderLeft: '3px solid var(--accent-cyan)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--accent-blue)' }}>
                  [{i + 1}] {chunk.title} / {chunk.section}
                </span>
                <span className="glass-pill" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                  sim: {(chunk.score * 100).toFixed(1)}%
                </span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5', margin: 0 }}>
                {chunk.text}
              </p>
            </div>
          ))}
          {!result && (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', fontStyle: 'italic', padding: '20px', textAlign: 'center' }}>
              Ask a question to inspect retrieved document chunks and similarity weights.
            </div>
          )}
        </div>

        {/* Document Ingestion section */}
        <div style={{ borderTop: '1px solid var(--glass-border-subtle)', paddingTop: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              INDEXED KNOWLEDGE BASE ({documents.length} DOCS)
            </span>
            <label className="glass-btn" style={{ padding: '4px 10px', fontSize: '0.72rem', cursor: 'pointer' }}>
              <Upload size={12} />
              <span>{uploading ? 'Indexing...' : 'Upload Doc'}</span>
              <input type="file" accept=".md,.txt" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </div>
          {uploadStatus && (
            <div style={{ fontSize: '0.72rem', color: 'var(--accent-emerald)', marginBottom: '6px' }}>
              {uploadStatus}
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {documents.map((d, i) => (
              <span key={i} className="glass-pill" style={{ fontSize: '0.7rem' }}>
                <FileText size={11} color="var(--accent-cyan)" />
                {d.title} ({d.sections} sections)
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
