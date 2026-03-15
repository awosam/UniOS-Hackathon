import React, { useState, useEffect } from 'react';
import { Brain, Calendar, CalendarDays, ListTodo, Send, Zap, X } from 'lucide-react';
import * as api from './api';
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
  const [events, setEvents] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [rightPanel, setRightPanel] = useState('dashboard'); // 'dashboard' | 'ongoing'
  const scrollRef = React.useRef(null);
  const swipeStartRef = React.useRef(null);

  useEffect(() => {
    fetchCanvasData();
    fetchEvents();
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

  const fetchEvents = async () => {
    try {
      const res = await api.getEvents();
      setEvents(Array.isArray(res.data?.events) ? res.data.events : []);
    } catch (e) {
      console.error("Events fetch failed");
    }
  };

  // Demo data when Canvas is not connected
  const fakeAssignments = [
    { name: 'Essay Draft', course: 'ENGL 109', due_at: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString() },
    { name: 'Problem Set 4', course: 'MATH 137', due_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() },
    { name: 'Midterm Quiz', course: 'CS 136', due_at: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(), isQuiz: true },
  ];
  const canvasItems = assignments.length > 0 ? assignments : fakeAssignments;

  const notifications = [
    'Exam schedule released',
    'Today is the last day to withdraw from the course',
  ];

  const pinnedEvents = [
    { title: 'Chess club at Hart House 5pm', startDate: null },
    { title: 'Intramurals cup starts next week Saturday the 21st', startDate: null },
  ];
  const allEvents = [...pinnedEvents, ...events].slice(0, 10);

  const ongoingTasks = [
    { title: 'Remember to apply for a minor degree', desc: 'You mentioned exploring a Philosophy minor last session', tag: 'Minor application' },
    { title: 'Follow up on your advisor meeting request', desc: 'Draft sent 3 days ago — no reply yet', tag: 'Pending' },
    { title: 'Check major switch GPA requirements', desc: 'You were researching CS to Data Science', tag: 'Major switch' },
  ];

  const handleRightPanelSwipeStart = (e) => {
    swipeStartRef.current = e.touches ? e.touches[0].clientX : e.clientX;
  };
  const handleRightPanelSwipeMove = (e) => { /* allow default scroll */ };
  const handleRightPanelSwipeEnd = (e) => {
    const endX = e.changedTouches ? e.changedTouches[0].clientX : e.clientX;
    const startX = swipeStartRef.current;
    if (startX == null) return;
    const delta = startX - endX;
    if (Math.abs(delta) < 50) return;
    if (delta > 0) setRightPanel('ongoing');
    else setRightPanel('dashboard');
    swipeStartRef.current = null;
  };
  const handleRightPanelMouseDown = (e) => {
    if (e.button !== 0) return;
    swipeStartRef.current = e.clientX;
  };
  const handleRightPanelMouseUp = (e) => {
    if (e.button !== 0 || swipeStartRef.current == null) return;
    const delta = swipeStartRef.current - e.clientX;
    if (Math.abs(delta) > 50) {
      if (delta > 0) setRightPanel('ongoing');
      else setRightPanel('dashboard');
    }
    swipeStartRef.current = null;
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

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === 'Escape') {
      setInput('');
      e.target.blur();
    }
  };

  const renderMessageContent = (msg) => {
    if (!msg.data) return <div className="markdown-wrapper"><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>;

    if (msg.dataType === 'pathfinder') {
      return (
        <div>
          <div className="markdown-wrapper" style={{ marginBottom: '12px' }}><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>
          <div className="pathfinder-steps">
            {msg.data.map((step, i) => (
              <div key={i} className="pathfinder-step-row">
                <span className="pathfinder-step-num">{i + 1}.</span>
                <span className="pathfinder-step-text">{step}</span>
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
          <pre className="drafter-preview">{msg.data}</pre>
        </div>
      );
    }

    return <div className="markdown-wrapper"><ReactMarkdown components={markdownComponents}>{msg.text}</ReactMarkdown></div>;
  };

  return (
    <div className="dashboard-container">
      {/* Left: Chat */}
      <main className="main-content">
        <div className="nb-window" style={{ flex: 1, minHeight: 0 }}>
          <div className="nb-title-bar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Brain size={22} strokeWidth={2.5} />
              <h1>Uni-OS</h1>
            </div>
            <span className="nb-title-dots" aria-hidden>⋯</span>
          </div>

          <div className="chat-panel-inner">
            <div className="chat-window">
              <div className="messages-container" ref={scrollRef} role="log" aria-live="polite" aria-label="Chat messages">
                {messages.map((msg, i) => (
                  <div key={i} className={`message ${msg.type}`} role={msg.type === 'user' ? 'article' : 'article'}>
                    {renderMessageContent(msg)}
                  </div>
                ))}
                {messages.length === 1 && (
                  <div className="chat-hint-below-first" aria-hidden>
                    use /finances, /schedule, or /anything to pull in personal context before sending
                  </div>
                )}
                {isLoading && (
                  <div className="message bot" aria-busy="true">
                    <p>Thinking...</p>
                  </div>
                )}
              </div>

              <div className="input-area">
                <input
                  className="chat-input"
                  placeholder="Ask me anything..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  aria-label="Chat message"
                  disabled={isLoading}
                />
                <button className="btn-primary" onClick={handleSend} disabled={isLoading} aria-label="Send message">
                  <Send size={20} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Right: Dashboard (swipe to Ongoing tasks) */}
      <aside
        className="context-pane nb-window"
        onTouchStart={handleRightPanelSwipeStart}
        onTouchEnd={handleRightPanelSwipeEnd}
        onMouseDown={handleRightPanelMouseDown}
        onMouseUp={handleRightPanelMouseUp}
      >
        <div className="nb-title-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h2>{rightPanel === 'dashboard' ? 'Dashboard' : 'Ongoing tasks'}</h2>
            <div className="panel-dots" aria-label="Switch panel">
              <button type="button" className={`panel-dot ${rightPanel === 'dashboard' ? 'active' : ''}`} aria-label="Dashboard" onClick={() => setRightPanel('dashboard')} />
              <button type="button" className={`panel-dot ${rightPanel === 'ongoing' ? 'active' : ''}`} aria-label="Ongoing tasks" onClick={() => setRightPanel('ongoing')} />
            </div>
          </div>
          <button type="button" aria-label="Close panel" style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
            <X size={20} strokeWidth={2.5} />
          </button>
        </div>

        <div className="right-panel-swipe">
          <div className={`right-panel-slider ${rightPanel === 'ongoing' ? 'panel-ongoing' : ''}`}>
            <div className="right-panel-pane">
              <div className="sidebar-inner">
                <section aria-labelledby="action-items-heading">
                  <h3 id="action-items-heading" className="nb-section-title">
                    <Calendar size={18} strokeWidth={2.5} />
                    Canvas
                  </h3>
                  <div className="deadlines-list">
                    {canvasItems.map((a, i) => (
                      <div key={i} className="deadline-card">
                        <h4>{a.name}{a.isQuiz ? ' (Quiz)' : ''}</h4>
                        <p>{a.course}</p>
                        <div className="deadline-card-meta">
                          <span className="deadline-card-date">{new Date(a.due_at).toLocaleDateString()}</span>
                          <Zap size={14} strokeWidth={2} />
                        </div>
                      </div>
                    ))}
                  </div>
                </section>

                <section aria-labelledby="notifications-heading">
                  <h3 id="notifications-heading" className="nb-section-title">Notifications</h3>
                  <ul className="deadlines-list" style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                    {notifications.map((text, i) => (
                      <li key={i} className="deadline-card" style={{ marginBottom: 8 }}>
                        <p style={{ margin: 0, fontSize: 14 }}>{text}</p>
                      </li>
                    ))}
                  </ul>
                </section>

                <section aria-labelledby="events-heading">
                  <h3 id="events-heading" className="nb-section-title">
                    <CalendarDays size={18} strokeWidth={2.5} />
                    Student life
                  </h3>
                  <div className="deadlines-list">
                    {allEvents.map((ev, i) => (
                      <div key={i} className="deadline-card">
                        <h4>{(ev.title || ev.name || 'Event').slice(0, 60)}{(ev.title || ev.name || '').length > 60 ? '…' : ''}</h4>
                        {ev.startDate ? (
                          <p className="deadline-card-date" style={{ marginTop: 4 }}>
                            {new Date(ev.startDate).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}
                          </p>
                        ) : null}
                        {ev.url && (
                          <a href={ev.url} target="_blank" rel="noopener noreferrer" className="citation-link" style={{ fontSize: 12, marginTop: 4, display: 'inline-block' }}>
                            More info
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </section>

                <div className="tip-card-wrap">
                  <div className="tip-card">
                    <p className="tip-card-label">Tip of the day</p>
                    <p className="tip-card-text">"Check the handbook for major switch rules early!"</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="right-panel-pane">
              <div className="sidebar-inner">
                <p className="ongoing-intro">Picked up from your previous chats</p>
                <section aria-labelledby="ongoing-tasks-heading">
                  <h3 id="ongoing-tasks-heading" className="nb-section-title">
                    <ListTodo size={18} strokeWidth={2.5} />
                    Ongoing tasks
                  </h3>
                  <div className="deadlines-list">
                    {ongoingTasks.map((task, i) => (
                      <div key={i} className="ongoing-task-card">
                        <h4>{task.title}</h4>
                        <p>{task.desc}</p>
                        <span className="ongoing-task-tag">{task.tag}</span>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="academics-section" aria-labelledby="academics-heading">
                  <h3 id="academics-heading" className="nb-section-title">Academics</h3>
                  <h4 className="nb-section-title" style={{ fontSize: 12, marginTop: 4 }}>Degree Registration Statuses</h4>
                  <div className="degree-card">
                    <div className="degree-row">
                      <span className="degree-name">UTSC Honours BA (Social Sci) 2022 Fall</span>
                      <span className="degree-status-tag">Registered</span>
                    </div>
                  </div>
                  <h4 className="nb-section-title" style={{ fontSize: 12, marginTop: 4 }}>Enrolled Courses: 2022 Fall</h4>
                  <div className="enrolled-courses-card">
                    <ul className="enrolled-courses-list" style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      <li className="enrolled-course-item">
                        <span className="course-code">ENG223H1 F</span> (First Sub-Session) Introduction to Shakespeare
                      </li>
                      <li className="enrolled-course-item">
                        <span className="course-code">ENG250H1 F</span> (Second Sub-Session) Introduction to American Literature
                      </li>
                      <li className="enrolled-course-item">
                        <span className="course-code">ENG203H1 F</span> (Second Sub-Session) British Literature in the World II: Romantic to Contemporary
                      </li>
                      <li className="enrolled-course-item">
                        <span className="course-code">ENG352H1 S</span> (Second Sub-Session) Modern Drama
                      </li>
                    </ul>
                  </div>
                </section>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
};

export default App;
