import { useState } from 'react'
import Sidebar from './components/Sidebar'
import CasePanel from './components/CasePanel'
import QueryMode from './components/modes/QueryMode'
import EvidenceMode from './components/modes/EvidenceMode'
import ContradictionsMode from './components/modes/ContradictionsMode'
import DevilsAdvocateMode from './components/modes/DevilsAdvocateMode'
import BriefMode from './components/modes/BriefMode'
import IngestPage from './components/pages/IngestPage'
import CasesPage from './components/pages/CasesPage'
import './App.css'

const MODES = [
  { id: 'query',          label: 'Query',            dot: '#7F77DD' },
  { id: 'evidence',       label: 'Evidence map',     dot: '#1D9E75' },
  { id: 'contradictions', label: 'Contradictions',   dot: '#e74c3c' },
  { id: 'devils',         label: "Devil's advocate", dot: '#f39c12' },
  { id: 'brief',          label: 'Trial brief',      dot: '#888780' },
]

export default function App() {
  const [page, setPage]     = useState('workspace')
  const [mode, setMode]     = useState('query')
  const [caseId, setCaseId] = useState('Case_001')

  const renderMain = () => {
    if (page === 'cases')  return <CasesPage onSelectCase={(id) => { setCaseId(id); setPage('workspace') }} />
    if (page === 'ingest') return <IngestPage caseId={caseId} setCaseId={setCaseId} />
    switch (mode) {
      case 'query':          return <QueryMode caseId={caseId} />
      case 'evidence':       return <EvidenceMode caseId={caseId} />
      case 'contradictions': return <ContradictionsMode caseId={caseId} />
      case 'devils':         return <DevilsAdvocateMode caseId={caseId} />
      case 'brief':          return <BriefMode caseId={caseId} />
      default:               return <QueryMode caseId={caseId} />
    }
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} />
      {page === 'workspace' && (
        <CasePanel
          caseId={caseId}
          setCaseId={setCaseId}
          modes={MODES}
          activeMode={mode}
          setMode={setMode}
        />
      )}
      <main className="main-area">{renderMain()}</main>
    </div>
  )
}
