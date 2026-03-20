import { useState, useRef, useEffect } from 'react'
import { query } from '../../api'
import './QueryMode.css'

const SUGGESTIONS = [
  'Summarise this case in 3 sentences',
  'What contradictions exist in the testimony?',
  'Who was present at the scene?',
  'What IPC sections could apply?',
  'Timeline of key events',
]

const MODES = ['evidence', 'opposition', 'assistant']

export default function QueryMode({ caseId }) {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const [queryMode, setQueryMode] = useState('evidence')
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text) => {
    const q = (text || input).trim()
    if (!q || !caseId.trim()) return

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setInput('')
    setLoading(true)

    try {
      const res = await query(caseId, q, queryMode)
      const d   = res.data
      setMessages(prev => [...prev, {
        role:       'assistant',
        content:    d.answer,
        citations:  d.citations,
        confidence: d.confidence,
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role:    'assistant',
        content: `Error: ${e.response?.data?.detail || e.message}`,
        error:   true,
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const formatCitations = (citations = []) => {
    return citations.map(c => c.citation_ref).filter(Boolean).join('  ·  ')
  }

  return (
    <div className="query-mode">
      <div className="query-topbar">
        <div>
          <h2 className="query-title">Query</h2>
          <p className="query-sub">{caseId}</p>
        </div>
        <div className="mode-pills">
          {MODES.map(m => (
            <button
              key={m}
              className={`mode-pill ${queryMode === m ? 'active' : ''}`}
              onClick={() => setQueryMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="chat-body">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">⚖</div>
            <h3 className="welcome-heading">How can I help with the case?</h3>
            <p className="welcome-sub">Query testimony · Cross-reference documents · Find contradictions</p>
            <div className="suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s} className="suggestion-pill" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
            {msg.citations?.length > 0 && (
              <div className="message-citations">
                {formatCitations(msg.citations)}
                {msg.confidence && ` · ${Math.round(msg.confidence * 100)}% confidence`}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="typing-dots">
              <span/><span/><span/>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask anything about the case..."
          rows={1}
          className="chat-textarea"
        />
        <button className="send-btn" onClick={() => send()} disabled={loading || !input.trim()}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="19" x2="12" y2="5"/>
            <polyline points="5 12 12 5 19 12"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
