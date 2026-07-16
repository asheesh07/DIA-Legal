import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  query, getChatSessions, getChatHistory,
  saveChatHistory, deleteChatSession,
  ingestPDFsStream, ingestVideoStream,
} from '@/api';
import { CitationBadge } from '@/components/shared/CitationBadge';
import { VideoCitationDialog } from '@/components/shared/VideoCitationDialog';
import { PdfCitationSheet } from '@/components/shared/PdfCitationSheet';
import { PrepareForTrialSheet } from './PrepareForTrialSheet';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';
import {
  Send, Paperclip, FileText, Gavel,
  MessageSquare, Plus, Trash, History,
} from 'lucide-react';
import {
  Sheet, SheetContent, SheetHeader, SheetTitle,
} from '@/components/ui/sheet';
import {
  Popover, PopoverContent, PopoverTrigger,
} from '@/components/ui/popover';
import { Video, File } from 'lucide-react';

// ── File limits ───────────────────────────────────────────────────
const PDF_MAX_FILES   = 5;
const PDF_MAX_BYTES   = 12  * 1024 * 1024;
const VIDEO_MAX_FILES = 2;
const VIDEO_MAX_BYTES = 250 * 1024 * 1024;

const STAGE_LABEL = {
  reading:          'Reading…',
  chunking:         'Chunking…',
  embedding:        'Embedding…',
  indexing:         'Indexing…',
  uploading:        'Uploading…',
  extracting_audio: 'Extracting audio…',
  transcribing:     'Transcribing…',
  diarizing:        'Identifying speakers…',
  sampling_frames:  'Sampling frames…',
};

// ── Suggestions ───────────────────────────────────────────────────
const SUGGESTIONS = [
  { icon: '🔍', text: 'Find any contradictions in these files' },
  { icon: '📋', text: 'Summarise this case in 3 sentences' },
  { icon: '⚖️', text: 'What will the other side argue against me?' },
  { icon: '🗂️', text: 'What key evidence do I have?' },
];

// ── No session selected ───────────────────────────────────────────
function NoSession({ onCreateCase }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
      <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
        <Gavel className="w-7 h-7 text-primary" />
      </div>
      <div>
        <h2 className="text-lg font-semibold">Welcome to DIA Legal</h2>
        <p className="text-sm text-muted-foreground mt-1 max-w-xs">
          Select a session from the sidebar or create a new one to get started.
        </p>
      </div>
      <Button onClick={onCreateCase}>
        <Plus className="w-4 h-4 mr-2" />
        New session
      </Button>
    </div>
  );
}

// ── Single message ────────────────────────────────────────────────
function Message({ msg, onCitationClick }) {
  const isUser = msg.role === 'user';
  return (
    <div className={cn('flex w-full', isUser ? 'justify-end' : 'justify-start')}>
      <div className={cn(
        'max-w-[80%] rounded-2xl px-4 py-3',
        isUser
          ? 'bg-primary text-primary-foreground rounded-br-none'
          : 'bg-card border border-border text-card-foreground rounded-bl-none shadow-sm',
      )}>
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
        {msg.citations?.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border/20 flex flex-wrap gap-2">
            {msg.citations.map((c, i) => (
              <CitationBadge key={i} citation={c} onClick={onCitationClick} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Typing indicator ──────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-card border border-border rounded-2xl rounded-bl-none px-4 py-3 shadow-sm">
        <div className="flex gap-1 items-center h-4">
          {[0, 150, 300].map(d => (
            <span key={d} className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce"
              style={{ animationDelay: `${d}ms` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main workspace ────────────────────────────────────────────────
export function Workspace({ caseId, sources = [], onIngested, onCreateCase }) {
  const [messages,            setMessages]            = useState([]);
  const [input,               setInput]               = useState('');
  const [loading,             setLoading]             = useState(false);
  const [sessions,            setSessions]            = useState([]);
  const [activeSessionId,     setActiveSessionId]     = useState(null);
  const [historyOpen,         setHistoryOpen]         = useState(false);
  const [trialOpen,           setTrialOpen]           = useState(false);
  const [activeVideoCitation, setActiveVideoCitation] = useState(null);
  const [activePdfCitation,   setActivePdfCitation]   = useState(null);

  const bottomRef   = useRef(null);
  const inputRef    = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    setMessages([]);
    setActiveSessionId(null);
    setSessions([]);
    if (!caseId) return;
    getChatSessions(caseId)
      .then(res => {
        const s = res.data?.sessions || [];
        setSessions(s);
        if (s.length > 0) setActiveSessionId(s[0].session_id);
      })
      .catch(console.error);
  }, [caseId]);

  useEffect(() => {
    if (!caseId || !activeSessionId) { setMessages([]); return; }
    getChatHistory(caseId, activeSessionId)
      .then(res => { if (res.data?.messages) setMessages(res.data.messages); })
      .catch(console.error);
  }, [caseId, activeSessionId]);

  const startNewChat = useCallback(() => {
    setActiveSessionId(null);
    setMessages([]);
    setHistoryOpen(false);
  }, []);

  const deleteSession = useCallback(async (sessionId, e) => {
    e.stopPropagation();
    try {
      await deleteChatSession(caseId, sessionId);
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
      if (activeSessionId === sessionId) { setActiveSessionId(null); setMessages([]); }
      toast.success('Chat deleted');
    } catch { toast.error('Failed to delete chat'); }
  }, [caseId, activeSessionId]);

  const send = useCallback(async (text) => {
    if (loading) return;
    const q = (text || input).trim();
    if (!q) return;
    if (!caseId) { toast.error('No session selected'); return; }

    let sid = activeSessionId;
    if (!sid) {
      sid = `chat_${Date.now()}`;
      setActiveSessionId(sid);
      setSessions(prev => [{ session_id: sid, title: q.slice(0, 40) }, ...prev]);
    }

    const title   = q.slice(0, 40);
    const history = messages.map(m => ({ role: m.role, content: m.content }));

    setMessages(prev => {
      const next = [...prev, { role: 'user', content: q }];
      saveChatHistory(caseId, sid, next, title).catch(console.error);
      return next;
    });
    setInput('');
    setLoading(true);

    try {
      const res = await query(caseId, q, 'evidence', history);
      const d   = res.data;
      setMessages(prev => {
        const next = [...prev, {
          role: 'assistant', content: d.answer,
          citations: d.citations, confidence: d.confidence,
        }];
        saveChatHistory(caseId, sid, next, title).catch(console.error);
        return next;
      });
    } catch (e) {
      setMessages(prev => {
        const next = [...prev, {
          role: 'assistant',
          content: `Error: ${e.response?.data?.detail || e.message}`,
          error: true,
        }];
        saveChatHistory(caseId, sid, next, title).catch(console.error);
        return next;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading, input, caseId, activeSessionId, messages]);

  // ── File ingest — triggered directly on file selection ───────────
  const handleFiles = useCallback(async (e) => {
    const picked = Array.from(e.target.files || []);
    e.target.value = '';
    if (!picked.length) return;
    if (!caseId) { toast.error('Select a session first'); return; }

    const isPdf  = f => f.name.toLowerCase().endsWith('.pdf');
    const pdfs   = picked.filter(isPdf);
    const videos = picked.filter(f => !isPdf(f));

    // Validate PDFs
    const pdfOk  = pdfs.filter(f => f.size <= PDF_MAX_BYTES).slice(0, PDF_MAX_FILES);
    pdfs.filter(f => f.size > PDF_MAX_BYTES).forEach(f =>
      toast.error(`"${f.name}" is too large — PDF files must be 12 MB or smaller`)
    );
    if (pdfs.length > PDF_MAX_FILES)
      toast.error(`You selected ${pdfs.length} PDFs — limit is ${PDF_MAX_FILES}. First ${PDF_MAX_FILES} used.`);

    // Validate videos
    const vidOk = videos.filter(f => f.size <= VIDEO_MAX_BYTES).slice(0, VIDEO_MAX_FILES);
    videos.filter(f => f.size > VIDEO_MAX_BYTES).forEach(f =>
      toast.error(`"${f.name}" is too large — video files must be 250 MB or smaller`)
    );
    if (videos.length > VIDEO_MAX_FILES)
      toast.error(`You selected ${videos.length} videos — limit is ${VIDEO_MAX_FILES}. First ${VIDEO_MAX_FILES} used.`);

    // Ingest PDFs as a batch
    if (pdfOk.length > 0) {
      const label = pdfOk.length === 1 ? `"${pdfOk[0].name}"` : `${pdfOk.length} PDFs`;
      const tid   = toast.loading(`Ingesting ${label}…`);
      try {
        await ingestPDFsStream(caseId, pdfOk, (ev) => {
          const stage = STAGE_LABEL[ev.stage];
          if (stage) toast.loading(`${label}: ${stage}`, { id: tid });
        });
        toast.success(`${label} indexed successfully`, { id: tid });
        onIngested?.();
      } catch (err) {
        toast.error(`${label}: ${err.message}`, { id: tid });
      }
    }

    // Ingest videos one at a time
    for (const vid of vidOk) {
      const tid = toast.loading(`Uploading "${vid.name}"…`);
      try {
        await ingestVideoStream(caseId, vid, (ev) => {
          const stage = STAGE_LABEL[ev.stage];
          if (stage) toast.loading(`"${vid.name}": ${stage}`, { id: tid });
        });
        toast.success(`"${vid.name}" indexed successfully`, { id: tid });
        onIngested?.();
      } catch (err) {
        toast.error(`"${vid.name}": ${err.message}`, { id: tid });
      }
    }
  }, [caseId, onIngested]);

  const handleCitationClick = useCallback((citation) => {
    const hasTime = Array.isArray(citation.time_range) && citation.time_range[1] > 0;
    const hasPage = citation.page_start > 0 || citation.page_end > 0;
    const type    = citation.type || (hasTime ? 'video' : hasPage ? 'pdf' : 'transcript');
    if (type === 'video')    setActiveVideoCitation(citation);
    else if (type === 'pdf') setActivePdfCitation(citation);
  }, []);

  if (!caseId) return <NoSession onCreateCase={onCreateCase} />;

  return (
    <div className="flex flex-col h-full">

      {/* ── Top bar ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
        <h1 className="font-semibold text-base truncate max-w-[280px]">{caseId}</h1>
        <div className="flex items-center gap-2">
          <Sheet open={historyOpen} onOpenChange={setHistoryOpen}>
            <Button variant="ghost" size="sm" className="text-muted-foreground"
              onClick={() => setHistoryOpen(true)}>
              <History className="w-4 h-4 mr-1.5" />History
            </Button>
            <SheetContent className="w-[280px]">
              <SheetHeader className="mb-4">
                <SheetTitle>Chat history</SheetTitle>
              </SheetHeader>
              <Button onClick={startNewChat} variant="outline" className="w-full mb-3" size="sm">
                <Plus className="w-4 h-4 mr-2" />New chat
              </Button>
              <div className="space-y-1">
                {sessions.length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-6">No chats yet.</p>
                )}
                {sessions.map(s => (
                  <div key={s.session_id}
                    className={cn(
                      'group flex items-center justify-between px-3 py-2 rounded-md cursor-pointer text-sm transition-colors',
                      activeSessionId === s.session_id ? 'bg-primary text-primary-foreground' : 'hover:bg-muted',
                    )}
                    onClick={() => { setActiveSessionId(s.session_id); setHistoryOpen(false); }}
                  >
                    <span className="truncate flex-1">{s.title || 'New chat'}</span>
                    <Button variant="ghost" size="icon"
                      className="w-5 h-5 opacity-0 group-hover:opacity-100 shrink-0 ml-1"
                      onClick={e => deleteSession(s.session_id, e)}>
                      <Trash className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </SheetContent>
          </Sheet>

          <Button onClick={() => setTrialOpen(true)} size="sm">
            <Gavel className="w-4 h-4 mr-2" />Prepare for Trial
          </Button>
        </div>
      </div>

      {/* ── Messages ────────────────────────────────────────────── */}
      <ScrollArea className="flex-1 px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center">
              <MessageSquare className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h2 className="font-semibold text-lg">How can I help with this case?</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Ask anything about your files, or pick a suggestion below.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} onClick={() => send(s.text)}
                  className="flex items-center gap-2.5 px-4 py-3 rounded-xl border border-border bg-card hover:bg-muted/60 transition-colors text-left text-sm">
                  <span className="text-base">{s.icon}</span>
                  <span className="text-muted-foreground leading-snug">{s.text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4 max-w-3xl mx-auto pb-4">
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} onCitationClick={handleCitationClick} />
            ))}
            {loading && <TypingIndicator />}
            <div ref={bottomRef} />
          </div>
        )}
      </ScrollArea>

      {/* ── Input area ──────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-border px-4 pt-3 pb-4 bg-background">
        <div className="max-w-3xl mx-auto space-y-2">
          <div className="flex items-end gap-2 border border-border rounded-xl bg-card px-3 py-2 shadow-sm focus-within:ring-1 focus-within:ring-primary transition-shadow">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask anything about your case… (Shift+Enter for new line)"
              className="min-h-[40px] max-h-[160px] border-0 shadow-none focus-visible:ring-0 resize-none py-1 px-0 text-sm flex-1 bg-transparent"
              rows={1}
            />
            <Button size="icon" className="shrink-0 h-8 w-8 rounded-lg mb-0.5"
              disabled={!input.trim() || loading} onClick={() => send()}>
              <Send className="w-3.5 h-3.5" />
            </Button>
          </div>

          {/* Hidden file input — accepts PDFs and common video formats */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.mp4,.mov,.mkv,.avi,.webm"
            multiple
            className="hidden"
            onChange={handleFiles}
          />

          <div className="flex items-center gap-3">
            <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5"
              onClick={() => fileInputRef.current?.click()}>
              <Paperclip className="w-3.5 h-3.5" />
              Add files
            </Button>

            {sources.length === 0 ? (
              <span className="text-xs text-muted-foreground flex items-center gap-1">
                <FileText className="w-3 h-3" />
                No files yet
              </span>
            ) : (
              <Popover>
                <PopoverTrigger asChild>
                  <button className="text-xs text-muted-foreground flex items-center gap-1 hover:text-foreground transition-colors">
                    <FileText className="w-3 h-3" />
                    {sources.length} file{sources.length !== 1 ? 's' : ''} in this session
                  </button>
                </PopoverTrigger>
                <PopoverContent side="top" align="start" className="w-72 p-2">
                  <p className="text-xs font-medium text-muted-foreground px-2 py-1 mb-1">
                    Files in this session
                  </p>
                  <div className="space-y-0.5 max-h-48 overflow-y-auto">
                    {sources.map((s, i) => {
                      const isVideo = s.type === 'video' || s.type === 'Youtube' || s.type === 'Local';
                      return (
                        <div key={i} className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-muted text-sm">
                          {isVideo
                            ? <Video className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                            : <File className="w-3.5 h-3.5 text-orange-500 shrink-0" />}
                          <span className="truncate flex-1 text-foreground">{s.name}</span>
                          <span className="text-xs text-muted-foreground shrink-0">{s.chunks} chunks</span>
                        </div>
                      );
                    })}
                  </div>
                </PopoverContent>
              </Popover>
            )}
          </div>
        </div>
      </div>

      {/* ── Sheets / dialogs ─────────────────────────────────────── */}
      <PrepareForTrialSheet caseId={caseId} open={trialOpen} onOpenChange={setTrialOpen} />
      <VideoCitationDialog citation={activeVideoCitation} isOpen={!!activeVideoCitation}
        onOpenChange={o => !o && setActiveVideoCitation(null)} />
      <PdfCitationSheet citation={activePdfCitation} isOpen={!!activePdfCitation}
        onOpenChange={o => !o && setActivePdfCitation(null)} />
    </div>
  );
}
