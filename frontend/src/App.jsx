import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ModelTabs from './components/ModelTabs';
import BenchmarkMatrix from './components/BenchmarkMatrix';

// 10 Model Views
import CharGptView from './components/models/CharGptView';
import RagView from './components/models/RagView';
import LlmView from './components/models/LlmView';
import WhisperView from './components/models/WhisperView';
import YoloView from './components/models/YoloView';
import BertView from './components/models/BertView';
import ClipView from './components/models/ClipView';
import LstmView from './components/models/LstmView';
import TwoTowerView from './components/models/TwoTowerView';
import MidiView from './components/models/MidiView';

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');
  const [activeTab, setActiveTab] = useState('benchmark');
  
  const [systemInfo, setSystemInfo] = useState(null);
  const [modelsData, setModelsData] = useState([]);
  const [benchmarks, setBenchmarks] = useState({});
  const [benchmarkingModel, setBenchmarkingModel] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Poll system stats and model registry info
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [sysRes, modelsRes, benchRes] = await Promise.all([
          fetch('/api/system'),
          fetch('/api/models'),
          fetch('/api/benchmarks')
        ]);
        if (sysRes.ok) setSystemInfo(await sysRes.json());
        if (modelsRes.ok) setModelsData(await modelsRes.json());
        if (benchRes.ok) {
          const bData = await benchRes.json();
          const cleanB = {};
          Object.entries(bData).forEach(([k, v]) => {
            if (v.result) cleanB[k] = v.result;
          });
          setBenchmarks(cleanB);
        }
      } catch (err) {
        // Backend warming up or disconnected
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleRunBenchmark = async (modelKey) => {
    setBenchmarkingModel(modelKey);
    try {
      const res = await fetch(`/api/models/${modelKey}/benchmark`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setBenchmarks(prev => ({ ...prev, [modelKey]: data }));
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBenchmarkingModel(null);
    }
  };

  const handleRunAllBenchmarks = async () => {
    setBenchmarkingModel('all');
    for (const m of modelsData) {
      try {
        const res = await fetch(`/api/models/${m.key}/benchmark`, { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          setBenchmarks(prev => ({ ...prev, [m.key]: data }));
        }
      } catch (e) {
        console.error(`Failed benchmarking ${m.key}:`, e);
      }
    }
    setBenchmarkingModel(null);
  };

  const handleLoadModel = async (modelKey) => {
    try {
      await fetch(`/api/models/${modelKey}/load`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  const handleUnloadModel = async (modelKey) => {
    try {
      await fetch(`/api/models/${modelKey}/unload`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  const currentModelInfo = modelsData.find(m => m.key === activeTab);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* Liquid Ambient Glowing Orbs */}
      <div 
        className="liquid-ambient-orb" 
        style={{ 
          top: '-10%', 
          left: '10%', 
          width: '550px', 
          height: '550px', 
          background: 'radial-gradient(circle, var(--accent-cyan) 0%, transparent 70%)' 
        }} 
      />
      <div 
        className="liquid-ambient-orb" 
        style={{ 
          top: '30%', 
          right: '-5%', 
          width: '650px', 
          height: '650px', 
          background: 'radial-gradient(circle, var(--accent-purple) 0%, transparent 70%)',
          animationDelay: '-7s'
        }} 
      />
      <div 
        className="liquid-ambient-orb" 
        style={{ 
          bottom: '-10%', 
          left: '30%', 
          width: '600px', 
          height: '600px', 
          background: 'radial-gradient(circle, var(--accent-blue) 0%, transparent 70%)',
          animationDelay: '-14s'
        }} 
      />

      {/* Glass Top Navigation */}
      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        systemInfo={systemInfo}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onRunAllBenchmarks={handleRunAllBenchmarks}
        isBenchmarkingAll={benchmarkingModel === 'all'}
      />

      {/* Model Selector Tabs */}
      <ModelTabs
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelsData={modelsData}
      />

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '0 24px 32px', position: 'relative', zIndex: 10 }}>
        {activeTab === 'benchmark' && (
          <BenchmarkMatrix
            modelsData={modelsData}
            benchmarks={benchmarks}
            onRunBenchmark={handleRunBenchmark}
            onRunAllBenchmarks={handleRunAllBenchmarks}
            benchmarkingModel={benchmarkingModel}
            onLoadModel={handleLoadModel}
            onUnloadModel={handleUnloadModel}
          />
        )}

        {activeTab === 'char_gpt' && (
          <CharGptView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'char_gpt'}
          />
        )}

        {activeTab === 'rag' && (
          <RagView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'rag'}
          />
        )}

        {activeTab === 'llm' && (
          <LlmView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'llm'}
          />
        )}

        {activeTab === 'whisper' && (
          <WhisperView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'whisper'}
          />
        )}

        {activeTab === 'yolo' && (
          <YoloView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'yolo'}
          />
        )}

        {activeTab === 'bert' && (
          <BertView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'bert'}
          />
        )}

        {activeTab === 'clip' && (
          <ClipView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'clip'}
          />
        )}

        {activeTab === 'lstm_ae' && (
          <LstmView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'lstm_ae'}
          />
        )}

        {activeTab === 'two_tower' && (
          <TwoTowerView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'two_tower'}
          />
        )}

        {activeTab === 'midi' && (
          <MidiView
            modelInfo={currentModelInfo}
            onRunBenchmark={handleRunBenchmark}
            isBenchmarking={benchmarkingModel === 'midi'}
          />
        )}
      </main>
    </div>
  );
}
