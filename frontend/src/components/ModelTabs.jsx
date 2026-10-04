import React from 'react';
import { 
  FileText, 
  Database, 
  Bot, 
  Mic, 
  Eye, 
  Smile, 
  Image as ImageIcon, 
  Activity, 
  Search, 
  Music,
  BarChart2,
  CheckCircle2,
  CircleDashed,
  AlertCircle
} from 'lucide-react';

export const MODEL_CONFIGS = [
  { key: 'char_gpt', name: 'Char-GPT', icon: FileText, modality: 'Text', color: '#38bdf8', tag: 'Text Gen' },
  { key: 'rag', name: 'RAG S-BERT', icon: Database, modality: 'Text', color: '#818cf8', tag: 'Knowledge' },
  { key: 'llm', name: 'Qwen/Llama+LoRA', icon: Bot, modality: 'Text', color: '#c084fc', tag: 'Chat / PEFT' },
  { key: 'whisper', name: 'Whisper ASR', icon: Mic, modality: 'Audio', color: '#f472b6', tag: 'Speech' },
  { key: 'yolo', name: 'YOLOv8', icon: Eye, modality: 'Vision', color: '#fbbf24', tag: 'Detection' },
  { key: 'bert', name: 'BERT Sentiment', icon: Smile, modality: 'Text', color: '#34d399', tag: 'Sentiment' },
  { key: 'clip', name: 'CLIP + Decoder', icon: ImageIcon, modality: 'Vision', color: '#2dd4bf', tag: 'Captioning' },
  { key: 'lstm_ae', name: 'LSTM Autoencoder', icon: Activity, modality: 'Series', color: '#f87171', tag: 'Anomaly' },
  { key: 'two_tower', name: 'Two-Tower Search', icon: Search, modality: 'Text', color: '#60a5fa', tag: 'Catalog ANN' },
  { key: 'midi', name: 'Music Transformer', icon: Music, modality: 'Music', color: '#a78bfa', tag: 'Symbolic MIDI' }
];

export default function ModelTabs({ activeTab, setActiveTab, modelsData }) {
  return (
    <nav style={{ 
      margin: '0 24px 20px', 
      display: 'flex', 
      gap: '8px', 
      overflowX: 'auto',
      paddingBottom: '8px',
      scrollbarWidth: 'thin'
    }}>
      {/* Benchmark Matrix Tab */}
      <button
        onClick={() => setActiveTab('benchmark')}
        className={`glass-card ${activeTab === 'benchmark' ? 'active-tab' : ''}`}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '10px 18px',
          borderRadius: 'var(--radius-sm)',
          cursor: 'pointer',
          border: activeTab === 'benchmark' ? '1px solid var(--accent-cyan)' : '1px solid var(--glass-border-subtle)',
          background: activeTab === 'benchmark' ? 'var(--glass-bg-active)' : 'var(--glass-card-bg)',
          boxShadow: activeTab === 'benchmark' ? 'var(--shadow-glow)' : 'none',
          whiteSpace: 'nowrap',
          transition: 'all 0.2s ease'
        }}
      >
        <BarChart2 size={18} color="var(--accent-cyan)" />
        <div style={{ textAlign: 'left' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700 }}>Evaluation Matrix</div>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Comparative Benchmarks</div>
        </div>
      </button>

      <div style={{ width: '1px', background: 'var(--glass-border)', margin: '4px 6px' }} />

      {/* 10 Models Tabs */}
      {MODEL_CONFIGS.map((m) => {
        const Icon = m.icon;
        const isActive = activeTab === m.key;
        const modelState = modelsData?.find(item => item.key === m.key);
        const status = modelState?.status || 'unloaded';

        return (
          <button
            key={m.key}
            onClick={() => setActiveTab(m.key)}
            className={`glass-card ${isActive ? 'active-tab' : ''}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '8px 16px',
              borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
              border: isActive ? `1px solid ${m.color}` : '1px solid var(--glass-border-subtle)',
              background: isActive ? 'var(--glass-bg-active)' : 'var(--glass-card-bg)',
              boxShadow: isActive ? `0 0 20px ${m.color}33` : 'none',
              whiteSpace: 'nowrap',
              transition: 'all 0.2s ease',
              position: 'relative'
            }}
          >
            <div style={{ 
              width: '30px', 
              height: '30px', 
              borderRadius: '8px', 
              background: `${m.color}22`,
              border: `1px solid ${m.color}55`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Icon size={16} color={m.color} />
            </div>

            <div style={{ textAlign: 'left' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>{m.name}</span>
                {status === 'ready' && (
                  <span title="Resident in Memory" style={{ display: 'inline-flex' }}>
                    <CheckCircle2 size={12} color="var(--accent-emerald)" />
                  </span>
                )}
                {status === 'loading' && (
                  <span title="Loading" className="spin" style={{ display: 'inline-flex' }}>
                    <CircleDashed size={12} color="var(--accent-amber)" />
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                <span>{m.tag}</span>
                {modelState?.avg_ms && (
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    · {modelState.avg_ms}ms
                  </span>
                )}
              </div>
            </div>
          </button>
        );
      })}
    </nav>
  );
}
