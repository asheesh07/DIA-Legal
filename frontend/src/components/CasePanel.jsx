import './CasePanel.css'

export default function CasePanel({ caseId, setCaseId, modes, activeMode, setMode }) {
  return (
    <div className="case-panel">
      <div className="case-panel-header">
        <p className="case-panel-label">Active case</p>
        <input
          value={caseId}
          onChange={e => setCaseId(e.target.value)}
          placeholder="Case_001"
          className="case-input"
        />
      </div>
      <nav className="mode-list">
        {modes.map(m => (
          <button
            key={m.id}
            className={`mode-item ${activeMode === m.id ? 'active' : ''}`}
            onClick={() => setMode(m.id)}
          >
            <span className="mode-dot" style={{ background: m.dot }} />
            <span className="mode-label">{m.label}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}
