import axios from 'axios'

const rawApiUrl = import.meta.env.VITE_API_URL;
const cleanApiUrl = rawApiUrl ? (rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl) : '';
const finalBaseUrl = cleanApiUrl
  ? (cleanApiUrl.endsWith('/api') ? cleanApiUrl : `${cleanApiUrl}/api`)
  : '/api';

const api = axios.create({
  baseURL: finalBaseUrl
})

export const getCases = () => api.get('/cases')
export const deleteCase = (id) => api.delete(`/cases/${id}`)
export const getCaseStats = (case_id) => api.get(`/cases/${case_id}/stats`)

// ── Legacy non-streaming ingest (kept for fallback) ──────────────
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

// ── SSE streaming ingest ─────────────────────────────────────────

const sseBase = finalBaseUrl;

/**
 * POST to a streaming endpoint and parse SSE events.
 * Calls onEvent(parsedData) for each event.
 * Returns a promise that resolves when the stream ends.
 */
async function _streamIngest(path, bodyOrFormData, onEvent, isForm = false) {
  const headers = isForm ? {} : { 'Content-Type': 'application/json' };
  const body    = isForm ? bodyOrFormData : JSON.stringify(bodyOrFormData);

  const res = await fetch(`${sseBase}${path}`, { method: 'POST', headers, body });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split('\n');
    buf = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        onEvent(data);
      } catch { /* ignore malformed */ }
    }
  }
}

export function ingestYoutubeStream(case_id, url, onEvent) {
  return _streamIngest('/ingest/youtube/stream', { case_id, url }, onEvent, false);
}

export function ingestVideoStream(case_id, file, onEvent) {
  const fd = new FormData();
  fd.append('case_id', case_id);
  fd.append('file', file);
  return _streamIngest('/ingest/video/stream', fd, onEvent, true);
}

export function ingestPDFsStream(case_id, files, onEvent) {
  const fd = new FormData();
  fd.append('case_id', case_id);
  files.forEach(f => fd.append('files', f));
  return _streamIngest('/ingest/pdf/stream', fd, onEvent, true);
}

// ── Query & chat ─────────────────────────────────────────────────
export const query = (case_id, query, mode, history = []) => api.post('/query', { case_id, query, mode, history })
export const getChatSessions = (case_id) => api.get(`/cases/${case_id}/chat/sessions`)
export const getChatHistory = (case_id, session_id) => api.get(`/cases/${case_id}/chat/sessions/${session_id}`)
export const saveChatHistory = (case_id, session_id, messages, title) => api.post(`/cases/${case_id}/chat/sessions/${session_id}`, { messages, session_id, title })
export const deleteChatSession = (case_id, session_id) => api.delete(`/cases/${case_id}/chat/sessions/${session_id}`)

export const evidenceMap = (case_id, lawyer_position) => api.post('/evidence-map', { case_id, lawyer_position })
export const detectContradictions = (case_id) => api.post('/contradictions', { case_id })

export const daNewSession = (case_id, topic) => api.post('/devils-advocate/session', { case_id, topic })
export const daArgue = (session_id, case_id, argument) => api.post('/devils-advocate/argue', { session_id, case_id, argument })
export const daSessions = (case_id) => api.get(`/devils-advocate/sessions/${case_id}`)

export const generateBrief = (case_id, lawyer_position) => api.post('/brief', { case_id, lawyer_position })
export const downloadBrief = (brief_id) => `/api/brief/download/${brief_id}`
