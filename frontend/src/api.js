import axios from 'axios'

const api = axios.create({ 
  baseURL: '/api'
})

export const getCases = () => api.get('/cases')
export const deleteCase = (id) => api.delete(`/cases/${id}`)

export const ingestYoutube = (case_id, url) => api.post('/ingest/youtube', { case_id, url })
export const ingestVideo = (case_id, file) => {
  const fd = new FormData()
  fd.append('case_id', case_id)
  fd.append('file', file)
  return api.post('/ingest/video', fd)
}
export const ingestPDFs = (case_id, files) => {
  const fd = new FormData()
  fd.append('case_id', case_id)
  files.forEach(f => fd.append('files', f))
  return api.post('/ingest/pdf', fd)
}

export const query = (case_id, query, mode) => api.post('/query', { case_id, query, mode })
export const evidenceMap = (case_id, lawyer_position) => api.post('/evidence-map', { case_id, lawyer_position })
export const detectContradictions = (case_id) => api.post('/contradictions', { case_id })

export const daNewSession = (case_id, topic) => api.post('/devils-advocate/session', { case_id, topic })
export const daArgue = (session_id, case_id, argument) => api.post('/devils-advocate/argue', { session_id, case_id, argument })
export const daSessions = (case_id) => api.get(`/devils-advocate/sessions/${case_id}`)

export const generateBrief = (case_id, lawyer_position) => api.post('/brief', { case_id, lawyer_position })
export const downloadBrief = (brief_id) => `/api/brief/download/${brief_id}`
