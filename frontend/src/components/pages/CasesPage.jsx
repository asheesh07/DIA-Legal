import { useState, useEffect } from 'react'
import { getCases, deleteCase } from '../../api'
import './CasesPage.css'

export default function CasesPage({ onSelectCase }) {
  const [cases, setCases]   = useState([])
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const res = await getCases()
      setCases(res.data.cases)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const remove = async (id, e) => {
    e.stopPropagation()
    if (!confirm(`Delete case ${id}?`)) return
    try {
      await deleteCase(id)
      setCases(prev => prev.filter(c => c.case_id !== id))
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div className="cases-page">
      <div className="cases-header">
        <h2 className="cases-title">Cases</h2>
        <button className="ghost-btn" onClick={load}>Refresh</button>
      </div>

      {loading ? (
        <p className="cases-loading">Loading...</p>
      ) : cases.length === 0 ? (
        <div className="cases-empty">
          <p>No cases yet. Ingest a video or PDF to create one.</p>
        </div>
      ) : (
        <div className="cases-list">
          {cases.map(c => (
            <div
              key={c.case_id}
              className="case-card"
              onClick={() => onSelectCase(c.case_id)}
            >
              <div className="case-card-main">
                <span className="case-card-id">{c.case_id}</span>
                <div className="case-card-meta">
                  <span>{c.source_count} source{c.source_count !== 1 ? 's' : ''}</span>
                  <span>·</span>
                  <span>{c.total_chunks} chunks</span>
                </div>
                <div className="case-sources">
                  {c.sources.slice(0, 3).map((s, i) => (
                    <span key={i} className="source-tag">{s.name.slice(0, 24)}</span>
                  ))}
                  {c.sources.length > 3 && (
                    <span className="source-tag muted">+{c.sources.length - 3} more</span>
                  )}
                </div>
              </div>
              <button
                className="case-delete-btn"
                onClick={e => remove(c.case_id, e)}
                title="Delete case"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
