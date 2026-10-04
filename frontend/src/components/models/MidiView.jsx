import React, { useState, useRef, useEffect } from 'react';
import { Music, Play, Square, Download, Sparkles, Volume2, Sliders, CheckCircle2 } from 'lucide-react';
import { playNotes, midiToFreq } from '../../utils/audioSynth';

const SCALES = ['major', 'minor', 'pentatonic', 'blues', 'dorian'];
const KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export default function MidiView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [scale, setScale] = useState('major');
  const [keyIndex, setKeyIndex] = useState(0); // C
  const [bars, setBars] = useState(8);
  const [temperature, setTemperature] = useState(0.9);
  const [bpm, setBpm] = useState(110);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeNote, setActiveNote] = useState(null);

  const canvasRef = useRef(null);
  const synthControllerRef = useRef(null);

  useEffect(() => {
    handleGenerate();
    return () => {
      synthControllerRef.current?.stop();
    };
  }, []);

  const handleGenerate = async () => {
    synthControllerRef.current?.stop();
    setIsPlaying(false);
    setActiveNote(null);
    setLoading(true);

    try {
      const res = await fetch('/api/midi/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scale,
          key: parseInt(keyIndex, 10),
          bars: parseInt(bars, 10),
          temperature: parseFloat(temperature),
          bpm: parseInt(bpm, 10)
        })
      });
      const data = await res.json();
      setResult(data);
      renderPianoRoll(data.notes);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePlayback = () => {
    if (isPlaying) {
      synthControllerRef.current?.stop();
      setIsPlaying(false);
      setActiveNote(null);
    } else if (result?.notes) {
      setIsPlaying(true);
      synthControllerRef.current = playNotes(result.notes, result.bpm || bpm, (note) => {
        setActiveNote(note);
        if (note === null) {
          setIsPlaying(false);
        }
      });
    }
  };

  const handleDownloadMidi = () => {
    if (!result?.midi_base64) return;
    const byteCharacters = atob(result.midi_base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neural_melody_${KEYS[keyIndex]}_${scale}.mid`;
    a.click();
  };

  const renderPianoRoll = (notes = result?.notes) => {
    if (!canvasRef.current || !notes || notes.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 240 * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = 240;
    ctx.clearRect(0, 0, W, H);

    const totalBeats = notes.reduce((max, n) => Math.max(max, n.start + n.dur), 0) || 16;
    const minPitch = 48; // C3
    const maxPitch = 84; // C6
    const pitchRange = maxPitch - minPitch;

    // Draw background keyboard rows
    for (let p = minPitch; p <= maxPitch; p++) {
      const isBlack = [1, 3, 6, 8, 10].includes(p % 12);
      const y = H - ((p - minPitch) / pitchRange) * H;
      ctx.fillStyle = isBlack ? 'rgba(0, 0, 0, 0.25)' : 'rgba(255, 255, 255, 0.03)';
      ctx.fillRect(0, y - H / pitchRange, W, H / pitchRange);
    }

    // Draw measure bar grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let b = 0; b <= totalBeats; b += 4) {
      const x = (b / totalBeats) * W;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    // Draw notes
    notes.forEach((note) => {
      const x = (note.start / totalBeats) * W;
      const w = Math.max(4, (note.dur / totalBeats) * W - 2);
      const y = H - ((note.pitch - minPitch) / pitchRange) * H;
      const h = Math.max(4, H / pitchRange - 1);

      const isActive = activeNote && activeNote.start === note.start && activeNote.pitch === note.pitch;

      if (isActive) {
        ctx.fillStyle = '#f472b6';
        ctx.shadowColor = '#f472b6';
        ctx.shadowBlur = 12;
      } else {
        ctx.fillStyle = 'var(--accent-purple)';
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      ctx.roundRect(x, y - h, w, h, 3);
      ctx.fill();
    });
    ctx.shadowBlur = 0;
  };

  useEffect(() => {
    renderPianoRoll();
  }, [activeNote]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
      {/* Left Column: Piano Roll Visualizer & Audio Player */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>Tiny Music Transformer</h2>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Symbolic Event-Based Autoregressive Generator
            </span>
          </div>
          <span className="glass-pill" style={{ color: 'var(--accent-purple)' }}>
            MIDI Synthesizer Active
          </span>
        </div>

        {/* Piano Roll Canvas */}
        <div style={{ 
          background: 'rgba(10, 15, 25, 0.65)', 
          borderRadius: 'var(--radius-md)', 
          padding: '12px',
          border: '1px solid var(--glass-border-subtle)',
          boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.5)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
            <span>KEYBOARD ROLL (C3 — C6)</span>
            <span>{result?.notes?.length || 0} NOTES GENERATED</span>
          </div>
          <canvas ref={canvasRef} style={{ width: '100%', height: '240px', display: 'block' }} />
        </div>

        {/* Synthesizer Transport Controls */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button
            className={`glass-btn ${isPlaying ? '' : 'glass-btn-primary'}`}
            onClick={handleTogglePlayback}
            disabled={!result?.notes || loading}
            style={{ flex: 1, padding: '12px' }}
          >
            {isPlaying ? <Square size={16} /> : <Play size={16} />}
            <span>{isPlaying ? 'Stop Synthesizer' : 'Play Melody in Browser'}</span>
          </button>

          <button
            className="glass-btn"
            onClick={handleDownloadMidi}
            disabled={!result?.midi_base64}
            style={{ padding: '12px 18px' }}
            title="Download standard MIDI file"
          >
            <Download size={16} />
            <span>.MID</span>
          </button>

          <button
            className="glass-btn glass-btn-primary"
            onClick={handleGenerate}
            disabled={loading}
            style={{ padding: '12px 20px' }}
          >
            <Sparkles size={16} className={loading ? 'spin' : ''} />
            <span>{loading ? 'Composing...' : 'Regenerate'}</span>
          </button>
        </div>
      </div>

      {/* Right Column: Musical Grammar Conditioning & Metrics */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Musical Grammar & Harmony</h3>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('midi')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run Music Benchmark'}
          </button>
        </div>

        {/* Scale & Key Selectors */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          <div>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              KEY ROOT
            </label>
            <select
              value={keyIndex}
              onChange={(e) => setKeyIndex(e.target.value)}
              className="glass-select"
            >
              {KEYS.map((k, i) => (
                <option key={k} value={i}>{k}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              MODAL SCALE
            </label>
            <select
              value={scale}
              onChange={(e) => setScale(e.target.value)}
              className="glass-select"
              style={{ textTransform: 'capitalize' }}
            >
              {SCALES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Sliders: Bars, Tempo, Temperature */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Phrase Length</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{bars} Bars</span>
            </div>
            <input
              type="range"
              min="4"
              max="16"
              step="4"
              value={bars}
              onChange={(e) => setBars(e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Tempo (BPM)</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{bpm} BPM</span>
            </div>
            <input
              type="range"
              min="70"
              max="160"
              step="5"
              value={bpm}
              onChange={(e) => setBpm(e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
              <span style={{ fontWeight: 600 }}>Creativity (Temperature)</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{temperature}</span>
            </div>
            <input
              type="range"
              min="0.4"
              max="1.4"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent-pink)' }}
            />
          </div>
        </div>

        {/* Music Theory Evaluation Metrics */}
        {result?.analysis && (
          <div className="glass-card" style={{ padding: '16px' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px' }}>
              THEORETICAL HARMONY ANALYSIS
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', textAlign: 'center' }}>
              <div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Scale Adherence</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-emerald)', fontSize: '1.1rem' }}>
                  {(result.analysis.in_scale_ratio * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Entropy</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem' }}>
                  {result.analysis.pitch_class_entropy}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Motif Diversity</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem' }}>
                  {(result.analysis.unique_trigram_ratio * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
