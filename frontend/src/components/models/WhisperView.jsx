import React, { useState, useRef } from 'react';
import { Mic, Upload, Play, Square, Volume2, Sparkles, CheckCircle2 } from 'lucide-react';

const SAMPLE_REFERENCE = "Welcome to Nimbus Corp. Please submit your expense claims within thirty days, and remember to enable multi factor authentication on your laptop.";

export default function WhisperView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [language, setLanguage] = useState('en');
  const [reference, setReference] = useState(SAMPLE_REFERENCE);
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
      };

      mediaRecorderRef.current.start();
      setRecording(true);
      setError(null);
    } catch (err) {
      setError('Microphone access denied: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      setRecording(false);
    }
  };

  const handleTranscribe = async (blobToUse = audioBlob) => {
    if (!blobToUse) return;
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append('file', blobToUse, 'audio.wav');
      fd.append('language', language);
      if (reference) fd.append('reference', reference);

      const res = await fetch('/api/whisper/transcribe', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSampleAudio = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/samples/speech_sample.wav');
      const blob = await res.blob();
      setAudioBlob(blob);
      setAudioUrl(URL.createObjectURL(blob));
      handleTranscribe(blob);
    } catch (e) {
      setError('Sample audio unavailable, record using mic or upload a .wav file.');
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '24px' }}>
      {/* Left Column: Audio Recording / Upload */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>Whisper Speech Recognition</h2>
            <span className="glass-pill" style={{ color: 'var(--accent-pink)' }}>
              OpenAI Whisper · Sequence-to-Sequence
            </span>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '6px' }}>
            Translates acoustic log-Mel spectrograms directly into punctuated text with timestamped chunks, Real-Time Factor (RTF), and Word Error Rate (WER) scoring.
          </p>
        </div>

        {/* Audio Input Options */}
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {!recording ? (
            <button
              className="glass-btn glass-btn-primary"
              onClick={startRecording}
              style={{ flex: 1, padding: '14px' }}
            >
              <Mic size={18} />
              <span>Record from Microphone</span>
            </button>
          ) : (
            <button
              className="glass-btn"
              onClick={stopRecording}
              style={{ flex: 1, padding: '14px', background: 'var(--accent-rose)', color: '#fff' }}
            >
              <Square size={18} />
              <span>Stop Recording...</span>
            </button>
          )}

          <button
            className="glass-btn"
            onClick={loadSampleAudio}
            style={{ padding: '14px 18px' }}
          >
            <Volume2 size={18} />
            <span>Load Sample Audio</span>
          </button>
        </div>

        {/* Audio Player */}
        {audioUrl && (
          <div className="glass-card" style={{ padding: '14px' }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
              RECORDED AUDIO PLAYBACK
            </label>
            <audio src={audioUrl} controls style={{ width: '100%', height: '36px' }} />
          </div>
        )}

        {/* Reference text for WER */}
        <div>
          <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
            GROUND TRUTH REFERENCE TEXT (FOR WORD ERROR RATE)
          </label>
          <textarea
            className="glass-textarea"
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            rows={2}
            placeholder="Type reference text to evaluate Word Error Rate..."
            style={{ fontSize: '0.85rem' }}
          />
        </div>

        {/* Language Selection */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>Target Language:</span>
          <select 
            value={language} 
            onChange={(e) => setLanguage(e.target.value)}
            className="glass-select"
            style={{ width: 'auto' }}
          >
            <option value="en">English (en)</option>
            <option value="es">Spanish (es)</option>
            <option value="fr">French (fr)</option>
            <option value="de">German (de)</option>
            <option value="auto">Auto-Detect</option>
          </select>
        </div>

        <button
          className="glass-btn glass-btn-primary"
          onClick={() => handleTranscribe()}
          disabled={loading || !audioBlob}
          style={{ padding: '12px', marginTop: 'auto' }}
        >
          <Sparkles size={16} className={loading ? 'spin' : ''} />
          <span>{loading ? 'Transcribing Spectrogram...' : 'Transcribe Audio'}</span>
        </button>
      </div>

      {/* Right Column: Transcription Output */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0 }}>Transcription & Timestamps</h3>
          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('whisper')}
            disabled={isBenchmarking}
            style={{ padding: '4px 10px', fontSize: '0.72rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run WER Benchmark'}
          </button>
        </div>

        {error && (
          <div className="glass-card" style={{ padding: '12px', borderLeft: '4px solid var(--accent-rose)', color: 'var(--accent-rose)', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}

        {/* Result & Metrics Banner */}
        {result && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
            <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Real-Time Factor</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {result.real_time_factor}x
              </div>
            </div>

            <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Duration</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {result.duration_s}s
              </div>
            </div>

            <div className="glass-card" style={{ padding: '10px', textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Word Error Rate</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                {result.wer !== undefined ? `${(result.wer * 100).toFixed(1)}%` : 'N/A'}
              </div>
            </div>
          </div>
        )}

        {/* Full Transcription text */}
        <div className="glass-card" style={{ 
          padding: '16px', 
          background: 'var(--glass-bg-active)',
          fontSize: '0.95rem',
          lineHeight: '1.6',
          fontWeight: 500,
          minHeight: '80px'
        }}>
          {result ? result.text : (
            <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontWeight: 400 }}>
              Transcribed text will appear here with high precision...
            </span>
          )}
        </div>

        {/* Timestamped Chunks */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '240px' }}>
          <label style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)' }}>
            TIMESTAMPED UTTERANCES
          </label>
          {result?.chunks?.map((c, i) => (
            <div key={i} className="glass-card" style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="glass-pill" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem' }}>
                {c.start?.toFixed(1)}s - {c.end?.toFixed(1)}s
              </span>
              <span style={{ fontSize: '0.85rem' }}>{c.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
