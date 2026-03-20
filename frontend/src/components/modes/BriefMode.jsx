import { useState } from 'react'
import { generateBrief, downloadBrief } from '../../api'
import './BriefMode.css'

const STRENGTH_COLOR = { STRONG: '#2ecc71', MODERATE: '#f39c12', WEAK: '#e74c3c' }
const RELIABILITY_COLOR = { FRIENDLY: '#2ecc71', NEUTRAL: '#f39c12', HOSTILE: '#e74c3c' }

export default function BriefMode({ caseId }) {
  const [position, setPosition] = useState('')
  const [brief, setBrief]       = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const run = async () => {
    if (!caseId.trim() || !position.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await generateBrief(caseId, position)
      setBrief(res.data)
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
          <h2 className="mode-title">Trial brief</h2>
          <p className="mode-sub">{caseId}</p>
        </div>
        {brief?.pdf_path && (
          <a
            href={downloadBrief(brief.brief_id)}
            className="download-btn"
            target="_blank"
            rel="noreferrer"
          >
            Download PDF
          </a>
        )}
      </div>

      <div className="brief-controls">
        <input
          value={position}
          onChange={e => setPosition(e.target.value)}
          placeholder="Your legal position..."
          className="position-input"
        />
        <button className="run-btn" onClick={run} disabled={loading || !position.trim()}>
          {loading ? 'Generating...' : 'Generate brief'}
        </button>
      </div>

      {error && <div className="error-bar">{error}</div>}

      {brief && (
        <div className="brief-body">
          <div className="brief-strength-bar">
            <span
              className="brief-strength-badge"
              style={{ color: STRENGTH_COLOR[brief.case_strength] || '#888' }}
            >
              {brief.case_strength} case
            </span>
            <span className="brief-id">Brief {brief.brief_id}</span>
          </div>

          <div className="brief-grid">
            <BriefSection title="Case summary">
              <p className="brief-text">{brief.case_summary}</p>
            </BriefSection>

            <BriefSection title="Overall assessment">
              <p className="brief-text">{brief.overall_assessment}</p>
            </BriefSection>

            <BriefSection title="Critical actions">
              <ul className="brief-list">
                {brief.critical_actions?.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </BriefSection>

            <BriefSection title="Cross-examination questions">
              <ol className="brief-list ordered">
                {brief.recommended_questions?.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ol>
            </BriefSection>
          </div>

          {brief.witness_profiles?.length > 0 && (
            <div className="brief-section-full">
              <p className="brief-section-title">Witness profiles</p>
              <div className="witness-grid">
                {brief.witness_profiles.map((w, i) => (
                  <div key={i} className="witness-card">
                    <div className="witness-header">
                      <span className="witness-id">{w.speaker_id}</span>
                      <span
                        className="witness-reliability"
                        style={{ color: RELIABILITY_COLOR[w.reliability_rating] || '#888' }}
                      >
                        {w.reliability_rating}
                      </span>
                    </div>
                    <p className="witness-role">{w.inferred_role}</p>
                    <div className="witness-stats">
                      <span>Credibility {w.credibility_score}/10</span>
                      <span>{w.contradiction_count} contradictions</span>
                    </div>
                    <p className="witness-approach">{w.recommended_approach}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {brief.contradictions?.length > 0 && (
            <div className="brief-section-full">
              <p className="brief-section-title">Contradictions</p>
              <div className="brief-contras">
                {brief.contradictions.map((c, i) => (
                  <div key={i} className="brief-contra-row">
                    <span
                      className="brief-contra-sev"
                      style={{ color: STRENGTH_COLOR[c.severity?.toUpperCase()] || '#888' }}
                    >
                      {c.severity}
                    </span>
                    <span className="brief-contra-cite">{c.citation_a}</span>
                    <span className="brief-contra-text">{c.statement_a?.slice(0, 80)}...</span>
                    <span className="brief-contra-vs">vs</span>
                    <span className="brief-contra-cite">{c.citation_b}</span>
                    <span className="brief-contra-text">{c.statement_b?.slice(0, 80)}...</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {!brief && !loading && !error && (
        <div className="empty-state">
          <p>Enter your legal position to generate a pre-trial intelligence report</p>
        </div>
      )}
    </div>
  )
}

function BriefSection({ title, children }) {
  return (
    <div className="brief-section">
      <p className="brief-section-title">{title}</p>
      {children}
    </div>
  )
}
