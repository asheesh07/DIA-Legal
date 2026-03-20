import { useState } from 'react'
import { ingestYoutube, ingestVideo, ingestPDFs } from '../../api'
import './IngestPage.css'

export default function IngestPage({ caseId, setCaseId }) {
  const [ytUrl, setYtUrl]         = useState('')
  const [videoFile, setVideoFile] = useState(null)
  const [pdfFiles, setPdfFiles]   = useState([])
  const [results, setResults]     = useState([])
  const [loading, setLoading]     = useState(false)

  const ingest = async () => {
    if (!caseId.trim()) return
    setLoading(true)
    setResults([])
    const newResults = []

    if (ytUrl.trim()) {
      try {
        const res = await ingestYoutube(caseId, ytUrl)
        newResults.push({ name: ytUrl.slice(0, 50), ...res.data })
      } catch (e) {
        newResults.push({ name: ytUrl.slice(0, 50), status: 'error', error: e.response?.data?.detail || e.message })
      }
    }

    if (videoFile) {
      try {
        const res = await ingestVideo(caseId, videoFile)
        newResults.push({ name: videoFile.name, ...res.data })
      } catch (e) {
        newResults.push({ name: videoFile.name, status: 'error', error: e.response?.data?.detail || e.message })
      }
    }

    if (pdfFiles.length > 0) {
      try {
        const res = await ingestPDFs(caseId, pdfFiles)
        res.data.results.forEach(r => newResults.push(r))
      } catch (e) {
        newResults.push({ name: 'PDFs', status: 'error', error: e.response?.data?.detail || e.message })
      }
    }

    setResults(newResults)
    setLoading(false)
  }

  const statusColor = (s) => s === 'success' ? '#2ecc71' : s === 'cached' ? '#f39c12' : '#e74c3c'
  const statusLabel = (s) => s === 'cached' ? 'already ingested' : s

  return (
    <div className="ingest-page">
      <div className="ingest-header">
        <h2 className="ingest-title">Ingest sources</h2>
        <p className="ingest-sub">Add video testimony and legal documents to a case</p>
      </div>

      <div className="ingest-form">
        <div className="ingest-field">
          <label className="field-label">Case ID</label>
          <input value={caseId} onChange={e => setCaseId(e.target.value)} placeholder="Case_001" />
        </div>

        <div className="ingest-divider" />

        <div className="ingest-field">
          <label className="field-label">YouTube URL</label>
          <input
            value={ytUrl}
            onChange={e => setYtUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
          />
        </div>

        <div className="ingest-field">
          <label className="field-label">Local video file</label>
          <div className="file-drop" onClick={() => document.getElementById('video-input').click()}>
            {videoFile ? (
              <span className="file-name">{videoFile.name}</span>
            ) : (
              <span className="file-placeholder">Click to select .mp4 .mov .mkv .avi</span>
            )}
          </div>
          <input
            id="video-input"
            type="file"
            accept=".mp4,.mov,.mkv,.avi,.webm"
            style={{ display: 'none' }}
            onChange={e => setVideoFile(e.target.files[0])}
          />
        </div>

        <div className="ingest-field">
          <label className="field-label">Legal documents — PDF</label>
          <div className="file-drop" onClick={() => document.getElementById('pdf-input').click()}>
            {pdfFiles.length > 0 ? (
              <span className="file-name">{pdfFiles.length} file{pdfFiles.length > 1 ? 's' : ''} selected</span>
            ) : (
              <span className="file-placeholder">Click to select PDFs (multiple allowed)</span>
            )}
          </div>
          <input
            id="pdf-input"
            type="file"
            accept=".pdf"
            multiple
            style={{ display: 'none' }}
            onChange={e => setPdfFiles(Array.from(e.target.files))}
          />
          {pdfFiles.length > 0 && (
            <div className="pdf-list">
              {pdfFiles.map((f, i) => (
                <span key={i} className="pdf-tag">{f.name}</span>
              ))}
            </div>
          )}
        </div>

        <button
          className="run-btn ingest-submit"
          onClick={ingest}
          disabled={loading || (!ytUrl.trim() && !videoFile && pdfFiles.length === 0)}
        >
          {loading ? 'Ingesting...' : 'Ingest sources'}
        </button>

        {results.length > 0 && (
          <div className="ingest-results">
            {results.map((r, i) => (
              <div key={i} className="ingest-result-row">
                <span className="result-name">{r.name}</span>
                <span className="result-status" style={{ color: statusColor(r.status) }}>
                  {statusLabel(r.status)}
                </span>
                {r.chunks && (
                  <span className="result-chunks">{r.chunks} chunks</span>
                )}
                {r.error && (
                  <span className="result-error">{r.error}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
