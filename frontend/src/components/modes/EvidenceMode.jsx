import { useState } from 'react'
import { evidenceMap } from '../../api'
import './EvidenceMode.css'

export default function EvidenceMode({ caseId }) {
  const [position, setPosition] = useState('')
  const [data, setData]         = useState(null)
  const [summary, setSummary]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const run = async () => {
    if (!caseId.trim() || !position.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await evidenceMap(caseId, position)
      setData(res.data)
      setSummary(res.data.summary)
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
          <h2 className="mode-title">Evidence map</h2>
          <p className="mode-sub">{caseId}</p>
        </div>
      </div>

      <div className="evidence-controls">
        <input
          value={position}
          onChange={e => setPosition(e.target.value)}
          placeholder="Your legal position — e.g. defending driver against negligence charges"
          className="position-input"
        />
        <button className="run-btn" onClick={run} disabled={loading || !position.trim()}>
          {loading ? 'Classifying...' : 'Map evidence'}
        </button>
      </div>

      {error && <div className="error-bar">{error}</div>}

      {summary && (
        <div className="summary-bar">
          <span className="sev-badge" style={{color:'#2ecc71'}}>{summary.supporting_count} supporting</span>
          <span className="sev-badge" style={{color:'#e74c3c'}}>{summary.opposing_count} opposing</span>
          <span className="sev-badge" style={{color:'#888780'}}>{summary.neutral_count} neutral</span>
          <span className="sev-badge muted">{summary.total_classified} total</span>
        </div>
      )}

      {data && (
        <div className="evidence-columns">
          <EvidenceColumn
            title="Supporting"
            color="#2ecc71"
            rows={data.supporting}
          />
          <EvidenceColumn
            title="Opposing"
            color="#e74c3c"
            rows={data.opposing}
          />
          <EvidenceColumn
            title="Neutral"
            color="#888780"
            rows={data.neutral}
          />
        </div>
      )}

      {!data && !loading && !error && (
        <div className="empty-state">
          <p>Enter your legal position to classify all case evidence</p>
        </div>
      )}
    </div>
  )
}

function EvidenceColumn({ title, color, rows }) {
  return (
    <div className="ev-col">
      <div className="ev-col-header" style={{ borderColor: color }}>
        <span className="ev-col-title" style={{ color }}>{title}</span>
        <span className="ev-col-count" style={{ color }}>{rows.length}</span>
      </div>
      <div className="ev-col-body">
        {rows.length === 0 && <p className="ev-empty">None found</p>}
        {rows.map((row, i) => (
          <div key={i} className="ev-card">
            <div className="ev-card-top">
              <span className="ev-strength">{row[0]}</span>
              <span className="ev-cite">{row[1]}</span>
            </div>
            <p className="ev-reason">{row[3]}</p>
            <p className="ev-text">{row[4]}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
