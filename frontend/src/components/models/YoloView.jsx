import React, { useState, useRef, useEffect } from 'react';
import { Eye, Upload, Sliders, Sparkles, CheckCircle2, Image as ImageIcon } from 'lucide-react';

const SAMPLES = [
  { name: 'City Street', file: 'street_city.jpg' },
  { name: 'Dog in Park', file: 'dog_park.jpg' },
  { name: 'Office Desk', file: 'office_desk.jpg' },
  { name: 'Cat on Sofa', file: 'cat_sofa.jpg' }
];

export default function YoloView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [selectedSample, setSelectedSample] = useState(SAMPLES[0]);
  const [imageSrc, setImageSrc] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const imageObjRef = useRef(null);

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
      detectImageBlob(blob);
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
    detectImageBlob(file);
  };

  const detectImageBlob = async (blob) => {
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', blob, 'image.jpg');
      fd.append('conf', conf);
      fd.append('iou', iou);

      const res = await fetch('/api/yolo/detect', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
      renderCanvas(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const renderCanvas = (detectionData) => {
    if (!imageSrc || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      const detections = detectionData?.detections || [];
      detections.forEach((det, idx) => {
        const [nx, ny, nw, nh] = det.box;
        const x = nx * img.width;
        const y = ny * img.height;
        const w = nw * img.width;
        const h = nh * img.height;

        // Vivid neon bounding box
        const colors = ['#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa', '#fb7185'];
        const color = colors[idx % colors.length];

        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, img.width / 350);
        ctx.strokeRect(x, y, w, h);

        // Glass badge label
        const labelText = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `bold ${Math.max(14, img.width / 50)}px sans-serif`;
        const textMetrics = ctx.measureText(labelText);
        const padding = 6;
        const tagH = Math.max(22, img.width / 40);

        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(x, Math.max(0, y - tagH), textMetrics.width + padding * 2, tagH);

        ctx.fillStyle = color;
        ctx.fillText(labelText, x + padding, Math.max(tagH - 6, y - 6));
      });
    };
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
      {/* Left Column: Canvas & Visual Overlay */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>YOLOv8 Object Detection</h2>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Ultralytics YOLOv8-nano · Single-Stage CNN
            </span>
          </div>
          <span className="glass-pill" style={{ color: 'var(--accent-amber)' }}>
            80 COCO Classes · Real-time
          </span>
        </div>

        {/* Canvas Display */}
        <div style={{ 
          position: 'relative', 
          width: '100%', 
          borderRadius: 'var(--radius-md)', 
          overflow: 'hidden',
          background: 'rgba(0,0,0,0.4)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '380px'
        }}>
          <canvas 
            ref={canvasRef} 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '440px', 
              objectFit: 'contain',
              borderRadius: 'var(--radius-md)'
            }} 
          />
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
                <Sparkles size={16} className="spin" /> Detecting Objects...
              </span>
            </div>
          )}
        </div>

        {/* Sample Gallery */}
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
                  border: selectedSample?.name === s.name ? '1px solid var(--accent-amber)' : '1px solid var(--glass-border-subtle)'
                }}
              >
                <ImageIcon size={14} />
                <span>{s.name}</span>
              </button>
            ))}

            <label className="glass-btn" style={{ padding: '6px 12px', fontSize: '0.78rem', cursor: 'pointer' }}>
              <Upload size={14} />
              <span>Upload Image</span>
              <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </div>
        </div>
      </div>

      {/* Right Column: Detections & Diagnostics */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Detections & Counts</h3>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('yolo')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run F1 Benchmark'}
          </button>
        </div>

        {/* Threshold Sliders */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Confidence Thr</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{(conf * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={conf}
              onChange={(e) => {
                setConf(parseFloat(e.target.value));
              }}
              style={{ width: '100%', accentColor: 'var(--accent-amber)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>NMS IoU Thr</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{(iou * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={iou}
              onChange={(e) => {
                setIou(parseFloat(e.target.value));
              }}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
            />
          </div>
        </div>

        {/* Classes Detected Chips */}
        {result?.counts && (
          <div>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
              DETECTED OBJECT CLASSES ({result.detections?.length || 0} TOTAL)
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {Object.entries(result.counts).map(([cls, cnt], i) => (
                <span key={i} className="glass-pill" style={{ 
                  fontSize: '0.8rem', 
                  padding: '6px 12px',
                  background: 'var(--accent-glow)',
                  borderColor: 'var(--accent-amber)',
                  fontWeight: 600
                }}>
                  {cls}: {cnt}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Inference Latency Breakdown */}
        {result?.speed_ms && (
          <div className="glass-card" style={{ padding: '16px' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
              PIPELINE LATENCY BREAKDOWN
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', textAlign: 'center' }}>
              <div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Preprocess</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{result.speed_ms.preprocess} ms</div>
              </div>
              <div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Inference</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-amber)' }}>{result.speed_ms.inference} ms</div>
              </div>
              <div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Postprocess (NMS)</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{result.speed_ms.postprocess} ms</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
