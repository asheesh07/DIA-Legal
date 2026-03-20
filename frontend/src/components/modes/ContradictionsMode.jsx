import { useState } from 'react'
import { detectContradictions } from '../../api'
import './ContradictionsMode.css'

const SEV_COLOR = { HIGH: '#e74c3c', MEDIUM: '#f39c12', LOW: '#2ecc71' }

export default function ContradictionsMode({ caseId }) {
  const [rows, setRows]       = useState([])
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const run = async () => {
    if (!caseId.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await detectContradictions(caseId)
      setRows(res.data.contradictions)
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
          <h2 className="mode-title">Contradictions</h2>
          <p className="mode-sub">{caseId}</p>
        </div>
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading ? 'Detecting...' : 'Detect contradictions'}
        </button>
      </div>

      {error && <div className="error-bar">{error}</div>}

      {summary && (
        <div className="summary-bar">
          <span className="sev-badge" style={{color: SEV_COLOR.HIGH}}>{summary.high_severity} high</span>
          <span className="sev-badge" style={{color: SEV_COLOR.MEDIUM}}>{summary.medium_severity} medium</span>
          <span className="sev-badge" style={{color: SEV_COLOR.LOW}}>{summary.low_severity} low</span>
          <span className="sev-badge">{summary.cross_source} cross-source</span>
          <span className="sev-badge muted">{summary.statements_checked} statements checked</span>
        </div>
      )}

      {rows.length > 0 ? (
        <div className="contradiction-list">
          {rows.map((row, i) => {
            const sev = row[0]?.replace(/[🔴🟡🟢⚪]/g, '').trim().toUpperCase()
            return (
              <div key={i} className="contradiction-card">
                <div className="contra-header">
                  <span className="contra-sev" style={{ color: SEV_COLOR[sev] || '#888' }}>
                    {row[0]}
                  </span>
                  {row[6] === '✅' && <span className="cross-badge">cross-source</span>}
                </div>
                <div className="contra-body">
                  <div className="contra-col">
                    <p className="contra-cite">{row[1]}</p>
                    <p className="contra-text">{row[2]}</p>
                  </div>
                  <div className="contra-vs">vs</div>
                  <div className="contra-col">
                    <p className="contra-cite">{row[3]}</p>
                    <p className="contra-text">{row[4]}</p>
                  </div>
                </div>
                <p className="contra-explanation">{row[5]}</p>
              </div>
            )
          })}
        </div>
      ) : (
        !loading && !error && (
          <div className="empty-state">
            <p>Run detection to find contradictions across all case sources</p>
          </div>
        )
      )}
    </div>
  )
}
