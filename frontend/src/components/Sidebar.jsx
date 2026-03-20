import './Sidebar.css'

const icons = {
  workspace: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  ),
  ingest: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  ),
  cases: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2"/>
      <path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/>
    </svg>
  ),
}

export default function Sidebar({ page, setPage }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span>DL</span>
      </div>
      {['workspace', 'ingest', 'cases'].map(p => (
        <button
          key={p}
          className={`sidebar-btn ${page === p ? 'active' : ''}`}
          onClick={() => setPage(p)}
          title={p.charAt(0).toUpperCase() + p.slice(1)}
        >
          {icons[p]}
        </button>
      ))}
    </aside>
  )
}
