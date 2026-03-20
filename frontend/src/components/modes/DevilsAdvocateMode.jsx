import { useState } from 'react'
import { daNewSession, daArgue } from '../../api'
import './DevilsAdvocateMode.css'

export default function DevilsAdvocateMode({ caseId }) {
  const [topic, setTopic]       = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [argument, setArgument] = useState('')
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')
  const [round, setRound]       = useState(0)

  const startSession = async () => {
    if (!caseId.trim() || !topic.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await daNewSession(caseId, topic)
      setSessionId(res.data.session_id)
      setResult(null)
      setRound(0)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const submitArgument = async () => {
    if (!argument.trim() || !sessionId) return
    setLoading(true)
    setError('')
    try {
      const res = await daArgue(sessionId, caseId, argument)
      setResult(res.data)
      setRound(res.data.round_number)
      setArgument('')
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mode-page">
      <div className="mode-topbar">
        <div>
          <h2 className="mode-title">Devil's advocate</h2>
          <p className="mode-sub">{caseId}{sessionId ? ` · Round ${round}` : ''}</p>
        </div>
      </div>

      {!sessionId ? (
        <div className="da-setup">
          <p className="da-setup-desc">
            Submit your legal arguments. The system challenges them using case evidence,
            finds weaknesses, and shows what the opposition will argue.
          </p>
          <div className="da-setup-form">
            <label className="field-label">Topic</label>
            <input
              value={topic}
              onChange={e => setTopic(e.target.value)}
              placeholder="e.g. Gearbox fault defence strategy"
            />
            <button
              className="run-btn"
              style={{ marginTop: 12 }}
              onClick={startSession}
              disabled={loading || !topic.trim()}
            >
              {loading ? 'Starting...' : 'Start session'}
            </button>
          </div>
          {error && <div className="error-bar">{error}</div>}
        </div>
      ) : (
        <div className="da-workspace">
          {error && <div className="error-bar" style={{margin:'12px 24px 0'}}>{error}</div>}

          {result && (
            <div className="da-results">
              <ResultBlock title="Overall critique" content={result.critique} />
              <div className="da-two-col">
                <ResultBlock title="Weaknesses found" content={result.weaknesses} />
                <ResultBlock title="Opposition will argue" content={result.opposition} />
              </div>
              <ResultBlock title="How to strengthen" content={result.strengthen} accent />
            </div>
          )}

          <div className="da-input-area">
            <label className="field-label">Your argument — Round {round + 1}</label>
            <textarea
              value={argument}
              onChange={e => setArgument(e.target.value)}
              placeholder="State your legal argument..."
              rows={4}
            />
            <div className="da-input-actions">
              <button
                className="ghost-btn"
                onClick={() => { setSessionId(null); setTopic(''); setResult(null) }}
              >
                New session
              </button>
              <button
                className="run-btn"
                onClick={submitArgument}
                disabled={loading || !argument.trim()}
              >
                {loading ? 'Analysing...' : 'Submit argument'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ResultBlock({ title, content, accent }) {
  return (
    <div className={`result-block ${accent ? 'accent' : ''}`}>
      <p className="result-block-title">{title}</p>
      <p className="result-block-content">{content}</p>
    </div>
  )
}
