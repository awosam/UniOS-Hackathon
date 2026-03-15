import React, { useState, useEffect } from 'react'; // React core hooks
import { motion, AnimatePresence } from 'framer-motion'; // Motion library for animations
import { 
  Brain, Calendar, FileText, Zap, User, 
  MapPin, Send, Plus, BookOpen, Settings,
  Activity, ArrowRight, ShieldCheck, Sparkles 
} from 'lucide-react'; // Icon library
import * as api from './api'; // Import our API utility layer
import ReactMarkdown from 'react-markdown';

const markdownComponents = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="citation-link">
      {children}
    </a>
  ),
};

const App = () => {
  const [messages, setMessages] = useState([
    { type: 'bot', text: "Hello! I'm Uni-OS, your academic companion. How can I help you today? (e.g., 'Switching majors', 'Canvas deadlines', 'Draft a request')" }
  ]);
  const [input, setInput] = useState('');
  const [assignments, setAssignments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = React.useRef(null);

  useEffect(() => {
    fetchCanvasData();
    // DEMO: Reset memory on reload for a fresh start
    api.resetMemory().catch(e => console.error("Memory reset failed"));
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchCanvasData = async () => {
    try {
      const res = await api.getAssignments();
      setAssignments(res.data.assignments);
    } catch (e) {
      console.error("Canvas fetch failed");
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { type: 'user', text: userMsg }]);
    setIsLoading(true);

    try {
      const res = await api.sendMessage(userMsg);
      setMessages(prev => [...prev, { 
        type: 'bot', 
        text: res.data.text, 
        data: res.data.data,
        dataType: res.data.type 
      }]);
    } catch (e) {
      const errorMsg = e.response?.data?.detail || "Is the backend running? I'm having trouble connecting to my brain.";
      setMessages(prev => [...prev, { type: 'bot', text: `Sorry, I hit a snag: ${errorMsg}` }]);
    }
    setIsLoading(false);
  };

  const renderMessageContent = (msg) => {
    if (!msg.data) return <div className="markdown-wrapper"><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>;

    if (msg.dataType === 'pathfinder') {
      return (
        <div>
          <div className="markdown-wrapper" style={{ marginBottom: '12px' }}><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '12px', border: '1px solid var(--border-light)' }}>
            {msg.data.map((step, i) => (
              <div key={i} style={{ display: 'flex', gap: '10px', marginBottom: '8px' }}>
                <span style={{ color: 'var(--primary)', fontWeight: 'bold' }}>{i + 1}.</span>
                <span style={{ fontSize: '14px' }}>{step}</span>
              </div>
            ))}
          </div>
        </div>
      );
    }

    if (msg.dataType === 'drafter') {
      return (
        <div>
          <div className="markdown-wrapper" style={{ marginBottom: '12px' }}><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>
          <pre style={{ 
            whiteSpace: 'pre-wrap', 
            background: 'rgba(0,0,0,0.2)', 
            padding: '16px', 
            borderRadius: '12px', 
            fontSize: '13px',
            fontFamily: 'monospace',
            border: '1px solid var(--border-light)'
          }}>
            {msg.data}
          </pre>
        </div>
      );
    }

    return <div className="markdown-wrapper"><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>;
  };


  return (
    <div className="dashboard-container">
      {/* Main Chat Section */}
      <main className="main-content" style={{ display: 'flex', flexDirection: 'column' }}>
        <header style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
          <div style={{ background: 'var(--primary)', padding: '10px', borderRadius: '12px' }}>
            <Brain size={28} color="white" />
          </div>
          <div>
            <h1 style={{ fontSize: '24px' }}>Uni-OS</h1>
            <p style={{ color: 'var(--text-dim)', fontSize: '14px' }}>Your academic lifestyle companion</p>
          </div>
        </header>

        <div className="chat-window">
          <div className="messages-container" ref={scrollRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.type}`}>
                {renderMessageContent(msg)}
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <p>Thinking...</p>
              </div>
            )}
          </div>

          <div className="input-area">
            <input 
              className="chat-input"
              placeholder="Ask me anything about your academic life..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            />
            <button className="btn-primary" onClick={handleSend} disabled={isLoading}>
              <Send size={20} />
            </button>
          </div>
        </div>
      </main>

      {/* Side Context Pane */}
      <aside className="context-pane">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
          <Calendar size={20} color="var(--primary)" />
          <h3 style={{ fontSize: '18px' }}>Action Items</h3>
        </div>

        <div className="deadlines-list">
          {assignments.length > 0 ? assignments.map((a, i) => (
            <div key={i} className="deadline-card">
              <h4>{a.name}</h4>
              <p>{a.course}</p>
              <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--accent)', fontSize: '11px', fontWeight: 'bold' }}>
                  {new Date(a.due_at).toLocaleDateString()}
                </span>
                <Zap size={14} color="var(--text-dim)" />
              </div>
            </div>
          )) : (
            <p style={{ color: 'var(--text-dim)', fontSize: '14px', textAlign: 'center', marginTop: '40px' }}>
              Connect Canvas to see your deadlines.
            </p>
          )}
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '40px' }}>
           <div className="glass-card" style={{ padding: '16px', textAlign: 'center' }}>
              <p style={{ fontSize: '13px', color: 'var(--text-dim)' }}>Tip of the day</p>
              <p style={{ fontSize: '14px', marginTop: '8px' }}>"Check the handbook for major switch rules early!"</p>
           </div>
        </div>
      </aside>
    </div>
  );
};

export default App;
