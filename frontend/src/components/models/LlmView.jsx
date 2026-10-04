import React, { useState, useRef, useEffect } from 'react';
import { Bot, Send, Sparkles, ToggleLeft, ToggleRight, Zap, RefreshCw } from 'lucide-react';

export default function LlmView({ modelInfo, onRunBenchmark, isBenchmarking }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am Nimbus Enterprise Assistant powered by Qwen 2.5 / Llama with parameter-efficient LoRA fine-tuning. How can I assist you with enterprise policies, code, or technical architecture?' }
  ]);
  const [input, setInput] = useState('');
  const [useLora, setUseLora] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [stats, setStats] = useState(null);
  
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || streaming) return;

    const userMsg = { role: 'user', content: input };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput('');
    setStreaming(true);
    setStats(null);

    // Placeholder for incoming assistant response
    const assistantIndex = newHistory.length;
    setMessages([...newHistory, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('/api/llm/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newHistory,
          max_new_tokens: 350,
          temperature: 0.7,
          use_lora: useLora
        })
      });

      if (!response.ok) throw new Error(await response.text());

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(Boolean);

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.type === 'token') {
              accumulated += data.text;
              setMessages(prev => {
                const copy = [...prev];
                copy[assistantIndex] = { role: 'assistant', content: accumulated };
                return copy;
              });
            } else if (data.type === 'done') {
              setStats(data);
            }
          } catch (err) {
            console.error('NDJSON parsing error:', err);
          }
        }
      }
    } catch (e) {
      setMessages(prev => {
        const copy = [...prev];
        copy[assistantIndex] = { role: 'assistant', content: `[Error: ${e.message}]` };
        return copy;
      });
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', height: '640px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>
              Qwen 2.5 / Llama 3.2 with LoRA
            </h2>
            <span className="glass-pill" style={{ color: 'var(--accent-purple)' }}>
              1.5B Params · Low-Rank Adaptation
            </span>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0' }}>
            Autoregressive causal LM streaming tokens in real time. Switch LoRA on/off for empirical A/B evaluation.
          </p>
        </div>

        {/* LoRA Toggle & Telemetry */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <button
            className="glass-card"
            onClick={() => setUseLora(!useLora)}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px', 
              padding: '6px 12px',
              cursor: 'pointer',
              border: useLora ? '1px solid var(--accent-purple)' : '1px solid var(--glass-border-subtle)'
            }}
          >
            {useLora ? <ToggleRight size={20} color="var(--accent-purple)" /> : <ToggleLeft size={20} color="var(--text-muted)" />}
            <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>
              LoRA Adapter: <strong style={{ color: useLora ? 'var(--accent-purple)' : 'var(--text-muted)' }}>{useLora ? 'ACTIVE' : 'BYPASSED'}</strong>
            </span>
          </button>

          <button
            className="glass-btn"
            onClick={() => onRunBenchmark('llm')}
            disabled={isBenchmarking}
            style={{ padding: '6px 12px', fontSize: '0.78rem' }}
          >
            {isBenchmarking ? 'Testing...' : 'Run QA Benchmark'}
          </button>
        </div>
      </div>

      {/* Chat Messages Log */}
      <div style={{ 
        flex: 1, 
        overflowY: 'auto', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '14px',
        padding: '16px',
        background: 'var(--glass-subtle)',
        borderRadius: 'var(--radius-md)',
        marginBottom: '16px'
      }}>
        {messages.map((m, i) => {
          const isUser = m.role === 'user';
          return (
            <div 
              key={i} 
              style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: isUser ? 'flex-end' : 'flex-start',
                maxWidth: '85%',
                alignSelf: isUser ? 'flex-end' : 'flex-start'
              }}
            >
              <div style={{ 
                fontSize: '0.72rem', 
                color: 'var(--text-muted)', 
                marginBottom: '4px',
                fontWeight: 600
              }}>
                {isUser ? 'YOU' : 'NIMBUS ASSISTANT'}
              </div>
              <div 
                className="glass-card" 
                style={{ 
                  padding: '12px 16px', 
                  borderRadius: 'var(--radius-sm)',
                  background: isUser ? 'var(--accent-gradient)' : 'var(--glass-card-bg)',
                  color: isUser ? '#ffffff' : 'var(--text-primary)',
                  fontSize: '0.9rem',
                  lineHeight: '1.5',
                  whiteSpace: 'pre-wrap',
                  boxShadow: isUser ? 'var(--shadow-glow)' : 'var(--shadow-sm)'
                }}
              >
                {m.content}
              </div>
            </div>
          );
        })}
        <div ref={chatEndRef} />
      </div>

      {/* Live Generation Telemetry Bar */}
      {stats && (
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          fontSize: '0.75rem', 
          color: 'var(--text-muted)', 
          fontFamily: 'var(--font-mono)',
          padding: '0 8px 10px'
        }}>
          <span>⚡ TTFT: {stats.ttft_ms} ms</span>
          <span>🚀 Throughput: {stats.tokens_per_sec} tok/s</span>
          <span>⏱️ Total: {stats.total_ms} ms</span>
          <span>💾 Memory: {stats.rss_mb} MB</span>
        </div>
      )}

      {/* Input Field */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          className="glass-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a technical or enterprise policy question..."
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={streaming}
        />
        <button
          className="glass-btn glass-btn-primary"
          onClick={handleSend}
          disabled={streaming || !input.trim()}
          style={{ padding: '0 24px' }}
        >
          <Send size={16} className={streaming ? 'spin' : ''} />
          <span>{streaming ? 'Generating...' : 'Send'}</span>
        </button>
      </div>
    </div>
  );
}
