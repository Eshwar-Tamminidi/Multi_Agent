import React, { useState, useRef, useEffect } from 'react';
import { Activity, RefreshCw, Sliders, AlertTriangle, CheckCircle2, Sparkles } from 'lucide-react';

export default function LstmView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [numPoints, setNumPoints] = useState(600);
  const [numAnomalies, setNumAnomalies] = useState(4);
  const [sensitivity, setSensitivity] = useState(1.0);
  const [seed, setSeed] = useState(42);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);

  useEffect(() => {
    runDetection();
  }, [sensitivity]);

  const runDetection = async (newSeed = seed) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/lstm_ae/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n: parseInt(numPoints, 10),
          anomalies: parseInt(numAnomalies, 10),
          sensitivity: parseFloat(sensitivity),
          seed: newSeed
        })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
      renderChart(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const renderChart = (data) => {
    if (!canvasRef.current || !data) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // High-DPI scaling
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 360 * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = 360;
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartW = W - padding.left - padding.right;
    const chartH = H - padding.top - padding.bottom;

    ctx.clearRect(0, 0, W, H);

    const vals = data.values || [];
    const recons = data.reconstruction || [];
    const flags = data.flags || [];
    const n = vals.length;
    if (n === 0) return;

    const minVal = Math.min(...vals, ...recons) - 5;
    const maxVal = Math.max(...vals, ...recons) + 5;
    const range = Math.max(1, maxVal - minVal);

    const getX = (i) => padding.left + (i / (n - 1)) * chartW;
    const getY = (v) => padding.top + chartH - ((v - minVal) / range) * chartH;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (i / 4) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(W - padding.right, y);
      ctx.stroke();

      const labelVal = (maxVal - (i / 4) * range).toFixed(0);
      ctx.fillStyle = 'rgba(150, 160, 180, 0.7)';
      ctx.font = '10px monospace';
      ctx.fillText(labelVal, 15, y + 4);
    }

    // Highlight Anomalous Regions
    ctx.fillStyle = 'rgba(251, 113, 133, 0.18)';
    let inAnom = false;
    let startX = 0;
    for (let i = 0; i < n; i++) {
      if (flags[i] && !inAnom) {
        inAnom = true;
        startX = getX(i);
      } else if (!flags[i] && inAnom) {
        inAnom = false;
        ctx.fillRect(startX, padding.top, getX(i) - startX, chartH);
      }
    }
    if (inAnom) {
      ctx.fillRect(startX, padding.top, getX(n - 1) - startX, chartH);
    }

    // Draw Reconstructed Normal Baseline (Subtle dashed purple line)
    ctx.strokeStyle = '#818cf8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = getX(i);
      const y = getY(recons[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Actual Telemetry Signal (Vivid Cyan line)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = getX(i);
      const y = getY(vals[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw red markers on anomalous points
    ctx.fillStyle = '#fb7185';
    for (let i = 0; i < n; i++) {
      if (flags[i]) {
        ctx.beginPath();
        ctx.arc(getX(i), getY(vals[i]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Panel */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>
                LSTM Autoencoder for Anomaly Detection
              </h2>
              <span className="glass-pill" style={{ color: 'var(--accent-rose)' }}>
                Recurrent Seq2Seq · 48-step Sliding Window
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0' }}>
              Detects server metric surges, network flatlines, noise bursts, and level shifts by measuring reconstruction divergence.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              className="glass-btn glass-btn-primary"
              onClick={() => {
                const newSeed = Math.floor(Math.random() * 10000);
                setSeed(newSeed);
                runDetection(newSeed);
              }}
              disabled={loading}
              style={{ fontSize: '0.82rem' }}
            >
              <RefreshCw size={15} className={loading ? 'spin' : ''} />
              <span>Simulate New Telemetry</span>
            </button>

            <button
              className="glass-btn"
              onClick={() => onRunBenchmark('lstm_ae')}
              disabled={isBenchmarking}
              style={{ fontSize: '0.82rem' }}
            >
              {isBenchmarking ? 'Testing...' : 'Run F1/AUC Benchmark'}
            </button>
          </div>
        </div>

        {/* Controls Bar */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '16px', 
          marginTop: '20px',
          padding: '16px',
          background: 'var(--glass-subtle)',
          borderRadius: 'var(--radius-sm)'
        }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Threshold Sensitivity</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{sensitivity}x</span>
            </div>
            <input
              type="range"
              min="0.4"
              max="2.5"
              step="0.1"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-rose)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Injected Anomalies</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{numAnomalies}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={numAnomalies}
              onChange={(e) => setNumAnomalies(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Series Length</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{numPoints} steps</span>
            </div>
            <input
              type="range"
              min="200"
              max="1500"
              step="100"
              value={numPoints}
              onChange={(e) => setNumPoints(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
          </div>
        </div>
      </div>

      {/* Chart Canvas Panel */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.8rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '14px', height: '3px', background: '#38bdf8', display: 'inline-block' }} />
              <span>Observed Signal (x)</span>
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '14px', height: '2px', borderTop: '2px dashed #818cf8', display: 'inline-block' }} />
              <span>LSTM Reconstruction (x̂)</span>
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '12px', height: '12px', background: 'rgba(251, 113, 133, 0.4)', borderRadius: '2px', display: 'inline-block' }} />
              <span>Flagged Anomaly Event</span>
            </span>
          </div>

          {result?.eval && (
            <div style={{ display: 'flex', gap: '10px' }}>
              <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                Precision: <strong>{(result.eval.precision * 100).toFixed(1)}%</strong>
              </span>
              <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                Recall: <strong>{(result.eval.recall * 100).toFixed(1)}%</strong>
              </span>
              <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent-emerald)' }}>
                F1 Score: <strong>{result.eval.f1}</strong>
              </span>
              {result.eval.roc_auc && (
                <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent-purple)' }}>
                  ROC-AUC: <strong>{result.eval.roc_auc}</strong>
                </span>
              )}
            </div>
          )}
        </div>

        <div style={{ width: '100%', position: 'relative' }}>
          <canvas ref={canvasRef} style={{ width: '100%', height: '360px', borderRadius: 'var(--radius-sm)' }} />
        </div>

        {/* Flagged Anomaly Events List */}
        {result?.events && (
          <div style={{ marginTop: '16px' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
              DETECTED INCIDENT WINDOWS ({result.events.length} EVENTS)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {result.events.map((ev, i) => (
                <div key={i} className="glass-card" style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem' }}>
                  <AlertTriangle size={13} color="var(--accent-rose)" />
                  <span>Window: <strong>t=[{ev.start}..{ev.end}]</strong></span>
                  <span style={{ color: 'var(--text-muted)' }}>· Peak MSE: {ev.peak_error}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
