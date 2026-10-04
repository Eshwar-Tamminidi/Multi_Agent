import React from 'react';
import { 
  Cpu, 
  HardDrive, 
  Layers, 
  Sun, 
  Moon, 
  Sparkles, 
  BarChart2, 
  Trash2,
  Activity,
  Zap
} from 'lucide-react';

export default function Navbar({ 
  theme, 
  onToggleTheme, 
  systemInfo, 
  activeTab, 
  setActiveTab, 
  onRunAllBenchmarks,
  isBenchmarkingAll
}) {
  return (
    <header className="glass-panel" style={{ 
      margin: '16px 24px', 
      padding: '12px 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '16px',
      zIndex: 50
    }}>
      {/* Brand & Identity */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{ 
          width: '42px', 
          height: '42px', 
          borderRadius: '12px', 
          background: 'var(--accent-gradient)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: 'var(--shadow-glow)'
        }}>
          <Sparkles size={22} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 style={{ 
              fontSize: '1.15rem', 
              fontWeight: 800, 
              letterSpacing: '-0.02em',
              background: 'var(--accent-gradient)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              margin: 0
            }}>
              Intelligent Multi-Modal Enterprise AI System
            </h1>
            <span className="glass-pill" style={{ fontSize: '0.7rem', padding: '2px 8px', color: 'var(--accent-cyan)' }}>
              v1.0 Pro
            </span>
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
            Lightweight Deep Learning & LLM Comparative Orchestration
          </p>
        </div>
      </div>

      {/* Live System Telemetry */}
      {systemInfo && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {/* Accelerator Device */}
          <div className="glass-pill" title="Inference Acceleration Device">
            <Zap size={14} color="var(--accent-amber)" />
            <span style={{ fontWeight: 600 }}>{systemInfo.device?.toUpperCase() || 'CPU'}</span>
            <span style={{ fontSize: '0.72rem', opacity: 0.8 }}>({systemInfo.platform})</span>
          </div>

          {/* Process Memory */}
          <div className="glass-pill" title="Current Python Process RSS Memory">
            <HardDrive size={14} color="var(--accent-cyan)" />
            <span>Process: <strong>{systemInfo.process_rss_mb} MB</strong></span>
          </div>

          {/* System RAM */}
          <div className="glass-pill" title="Host Physical RAM Usage">
            <Cpu size={14} color="var(--accent-purple)" />
            <span>RAM: <strong>{systemInfo.ram_used_gb} / {systemInfo.ram_total_gb} GB</strong> ({systemInfo.ram_percent}%)</span>
          </div>

          {/* Loaded Models */}
          <div className="glass-pill" title="Models Resident in Memory">
            <Layers size={14} color="var(--accent-emerald)" />
            <span>Loaded: <strong>{systemInfo.loaded?.length || 0} / 10</strong></span>
          </div>
        </div>
      )}

      {/* Global Actions & Theme Switcher */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <button 
          className={`glass-btn ${activeTab === 'benchmark' ? 'glass-btn-primary' : ''}`}
          onClick={() => setActiveTab('benchmark')}
          style={{ padding: '8px 14px', fontSize: '0.82rem' }}
        >
          <BarChart2 size={16} />
          <span>Evaluation Matrix</span>
        </button>

        <button 
          className="glass-btn" 
          onClick={onToggleTheme}
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Theme`}
          style={{ padding: '8px 12px' }}
        >
          {theme === 'dark' ? <Sun size={17} color="var(--accent-amber)" /> : <Moon size={17} color="var(--accent-indigo)" />}
        </button>
      </div>
    </header>
  );
}
