import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, Upload, Sparkles, Tag, CheckCircle2 } from 'lucide-react';

const SAMPLES = [
  { name: 'Dog in Park', file: 'dog_park.jpg' },
  { name: 'City Street', file: 'street_city.jpg' },
  { name: 'Office Desk', file: 'office_desk.jpg' },
  { name: 'Cat on Sofa', file: 'cat_sofa.jpg' }
];

export default function ClipView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [selectedSample, setSelectedSample] = useState(SAMPLES[0]);
  const [imageSrc, setImageSrc] = useState(null);
  const [customLabels, setCustomLabels] = useState('golden retriever, dog playing, modern office, busy downtown, animal');
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadSample(SAMPLES[0]);
  }, []);

  const loadSample = async (sample) => {
    setSelectedSample(sample);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/samples/${sample.file}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setImageSrc(url);
      analyzeImageBlob(blob);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageSrc(url);
    setSelectedSample(null);
    analyzeImageBlob(file);
  };

  const analyzeImageBlob = async (blob) => {
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', blob, 'image.jpg');
      if (customLabels) fd.append('labels', customLabels);

      const res = await fetch('/api/clip/analyze', { method: 'POST', body: fd });
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
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '24px' }}>
      {/* Left Column: Image Viewer & Gallery */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>CLIP + Transformer Decoder</h2>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              OpenAI ViT-B/32 + 3L Prefix Captioning Decoder
            </span>
          </div>
          <span className="glass-pill" style={{ color: 'var(--accent-cyan)' }}>
            Vision-Language Fusion
          </span>
        </div>

        {/* Image Preview Container */}
        <div style={{ 
          width: '100%', 
          height: '340px', 
          borderRadius: 'var(--radius-md)', 
          overflow: 'hidden',
          background: 'rgba(0,0,0,0.3)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          position: 'relative'
        }}>
          {imageSrc ? (
            <img 
              src={imageSrc} 
              alt="Analyzed preview" 
              style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
            />
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>No image loaded</span>
          )}

          {loading && (
            <div style={{ 
              position: 'absolute', 
              inset: 0, 
              background: 'rgba(0,0,0,0.5)', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              backdropFilter: 'blur(4px)'
            }}>
              <span className="glass-pill" style={{ fontSize: '0.9rem', padding: '8px 16px' }}>
                <Sparkles size={16} className="spin" /> Decoding Multi-Modal Embedding...
              </span>
            </div>
          )}
        </div>

        {/* Gallery buttons */}
        <div>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            IMAGE GALLERY
          </label>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {SAMPLES.map((s, i) => (
              <button
                key={i}
                className="glass-btn"
                onClick={() => loadSample(s)}
                style={{
                  padding: '6px 12px',
                  fontSize: '0.78rem',
                  border: selectedSample?.name === s.name ? '1px solid var(--accent-cyan)' : '1px solid var(--glass-border-subtle)'
                }}
              >
                <ImageIcon size={14} />
                <span>{s.name}</span>
              </button>
            ))}

            <label className="glass-btn" style={{ padding: '6px 12px', fontSize: '0.78rem', cursor: 'pointer' }}>
              <Upload size={14} />
              <span>Upload Custom</span>
              <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </div>
        </div>

        {/* Candidate labels input */}
        <div>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
            CUSTOM ZERO-SHOT CLASSIFICATION LABELS (COMMA SEPARATED)
          </label>
          <input
            type="text"
            className="glass-input"
            value={customLabels}
            onChange={(e) => setCustomLabels(e.target.value)}
            placeholder="e.g. dog, cat, modern office, street, car"
          />
        </div>
      </div>

      {/* Right Column: Caption, Tags & Alignment */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Synthesized Understanding</h3>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('clip')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run Top-5 Benchmark'}
          </button>
        </div>

        {result && (
          <>
            {/* Generated Caption Card */}
            <div className="glass-card" style={{ padding: '18px', background: 'var(--glass-bg-active)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span className="metric-lbl">AUTONOMOUS IMAGE CAPTION</span>
                <span className="glass-pill" style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)' }}>
                  CLIP Score: {result.clip_score}%
                </span>
              </div>
              <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: '1.5' }}>
                "{result.caption}"
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
                Encoder: {result.encoder_ms} ms · Decoder: {result.decoder_ms} ms
              </div>
            </div>

            {/* Zero-Shot Label Ranking */}
            {result.zero_shot && (
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
                  ZERO-SHOT MULTI-MODAL PROBABILITIES
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {result.zero_shot.map((zs, i) => (
                    <div key={i} style={{ fontSize: '0.78rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                        <span style={{ fontWeight: 600 }}>{zs.label}</span>
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{(zs.score * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '7px', background: 'var(--glass-subtle)', borderRadius: '999px', overflow: 'hidden' }}>
                        <div style={{ 
                          width: `${zs.score * 100}%`, 
                          height: '100%', 
                          background: 'linear-gradient(90deg, #38bdf8, #818cf8)',
                          borderRadius: '999px'
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top Concept Tags */}
            {result.tags && (
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
                  EMERGENT CONCEPT EMBEDDINGS
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {result.tags.map((t, i) => (
                    <span key={i} className="glass-pill" style={{ fontSize: '0.75rem', padding: '6px 12px' }}>
                      <Tag size={12} color="var(--accent-cyan)" />
                      <span>{t.label}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', opacity: 0.8 }}>{(t.score * 100).toFixed(0)}%</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {!result && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', fontStyle: 'italic', textAlign: 'center', padding: '40px' }}>
            Select an image from the gallery to run CLIP multi-modal embedding & autoregressive caption generation.
          </div>
        )}
      </div>
    </div>
  );
}
