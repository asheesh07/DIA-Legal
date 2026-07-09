import React, { useState, useRef, useEffect, useCallback } from 'react';
import { query, getChatSessions, getChatHistory, saveChatHistory, deleteChatSession } from '@/api';
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
  Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger,
} from '@/components/ui/sheet';

// ── Suggestions shown on an empty session ─────────────────────────
const SUGGESTIONS = [
  { icon: '🔍', text: 'Find any contradictions in these files' },
  { icon: '📋', text: 'Summarise this case in 3 sentences' },
  { icon: '⚖️', text: 'What will the other side argue against me?' },
  { icon: '🗂️', text: 'What key evidence do I have?' },
];

// ── Empty state — no session selected ─────────────────────────────
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

// ── Single chat message ────────────────────────────────────────────
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

// ── Typing indicator ───────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-card border border-border rounded-2xl rounded-bl-none px-4 py-3 shadow-sm">
        <div className="flex gap-1 items-center h-4">
          {[0, 150, 300].map(delay => (
            <span
              key={delay}
              className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main workspace ─────────────────────────────────────────────────
export function Workspace({ caseId, fileCount, onAddFiles, onCreateCase }) {
  const [messages,         setMessages]         = useState([]);
  const [input,            setInput]            = useState('');
  const [loading,          setLoading]          = useState(false);
  const [sessions,         setSessions]         = useState([]);
  const [activeSessionId,  setActiveSessionId]  = useState(null);
  const [historyOpen,      setHistoryOpen]      = useState(false);
  const [trialOpen,        setTrialOpen]        = useState(false);
  const [activeVideoCitation, setActiveVideoCitation] = useState(null);
  const [activePdfCitation,   setActivePdfCitation]   = useState(null);

  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Reset + load sessions when case changes
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
      .catch(e => console.error('Failed to load sessions:', e));
  }, [caseId]);

  // Load messages when session changes
  useEffect(() => {
    if (!caseId || !activeSessionId) { setMessages([]); return; }
    getChatHistory(caseId, activeSessionId)
      .then(res => { if (res.data?.messages) setMessages(res.data.messages); })
      .catch(e => console.error('Failed to load history:', e));
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
    } catch {
      toast.error('Failed to delete chat');
    }
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

    const title = q.slice(0, 40);
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
          role: 'assistant',
          content: d.answer,
          citations: d.citations,
          confidence: d.confidence,
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

  const handleCitationClick = useCallback((citation) => {
    const hasTime = Array.isArray(citation.time_range) && citation.time_range[1] > 0;
    const hasPage = citation.page_start > 0 || citation.page_end > 0;
    const type = citation.type || (hasTime ? 'video' : hasPage ? 'pdf' : 'transcript');
    if (type === 'video')     setActiveVideoCitation(citation);
    else if (type === 'pdf')  setActivePdfCitation(citation);
  }, []);

  // ── No session selected ─────────────────────────────────────────
  if (!caseId) return <NoSession onCreateCase={onCreateCase} />;

  const fileLabel = fileCount > 0
    ? `${fileCount} file${fileCount !== 1 ? 's' : ''} in this session`
    : 'No files yet';

  return (
    <div className="flex flex-col h-full">

      {/* ── Top bar ──────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="font-semibold text-base truncate max-w-[280px]">{caseId}</h1>
        </div>
        <div className="flex items-center gap-2">
          {/* History */}
          <Sheet open={historyOpen} onOpenChange={setHistoryOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="sm" className="text-muted-foreground">
                <History className="w-4 h-4 mr-1.5" />
                History
              </Button>
            </SheetTrigger>
            <SheetContent className="w-[280px]">
              <SheetHeader className="mb-4">
                <SheetTitle>Chat history</SheetTitle>
              </SheetHeader>
              <Button onClick={startNewChat} variant="outline" className="w-full mb-3" size="sm">
                <Plus className="w-4 h-4 mr-2" />
                New chat
              </Button>
              <div className="space-y-1">
                {sessions.length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-6">No chats yet.</p>
                )}
                {sessions.map(s => (
                  <div
                    key={s.session_id}
                    className={cn(
                      'group flex items-center justify-between px-3 py-2 rounded-md cursor-pointer text-sm transition-colors',
                      activeSessionId === s.session_id
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-muted',
                    )}
                    onClick={() => { setActiveSessionId(s.session_id); setHistoryOpen(false); }}
                  >
                    <span className="truncate flex-1">{s.title || 'New chat'}</span>
                    <Button
                      variant="ghost" size="icon"
                      className="w-5 h-5 opacity-0 group-hover:opacity-100 shrink-0 ml-1"
                      onClick={e => deleteSession(s.session_id, e)}
                    >
                      <Trash className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </SheetContent>
          </Sheet>

          {/* Prepare for Trial */}
          <Button onClick={() => setTrialOpen(true)} size="sm">
            <Gavel className="w-4 h-4 mr-2" />
            Prepare for Trial
          </Button>
        </div>
      </div>

      {/* ── Messages ─────────────────────────────────────────────── */}
      <ScrollArea className="flex-1 px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center">
              <MessageSquare className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h2 className="font-semibold text-lg">How can I help with this case?</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Ask me anything about your files, or pick a suggestion below.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => send(s.text)}
                  className="flex items-center gap-2.5 px-4 py-3 rounded-xl border border-border bg-card hover:bg-muted/60 transition-colors text-left text-sm"
                >
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

      {/* ── Input area ───────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-border px-4 pt-3 pb-4 bg-background">
        <div className="max-w-3xl mx-auto space-y-2">
          {/* Input box */}
          <div className="flex items-end gap-2 border border-border rounded-xl bg-card px-3 py-2 shadow-sm focus-within:ring-1 focus-within:ring-primary transition-shadow">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
              }}
              placeholder="Ask anything about your case… (Shift+Enter for new line)"
              className="min-h-[40px] max-h-[160px] border-0 shadow-none focus-visible:ring-0 resize-none py-1 px-0 text-sm flex-1 bg-transparent"
              rows={1}
            />
            <Button
              size="icon"
              className="shrink-0 h-8 w-8 rounded-lg mb-0.5"
              disabled={!input.trim() || loading}
              onClick={() => send()}
            >
              <Send className="w-3.5 h-3.5" />
            </Button>
          </div>

          {/* Add files + file count */}
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={onAddFiles}
              className="h-8 text-xs gap-1.5"
            >
              <Paperclip className="w-3.5 h-3.5" />
              Add files
            </Button>
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <FileText className="w-3 h-3" />
              {fileLabel}
            </span>
          </div>
        </div>
      </div>

      {/* ── Modals ───────────────────────────────────────────────── */}
      <PrepareForTrialSheet
        caseId={caseId}
        open={trialOpen}
        onOpenChange={setTrialOpen}
      />
      <VideoCitationDialog
        citation={activeVideoCitation}
        isOpen={!!activeVideoCitation}
        onOpenChange={o => !o && setActiveVideoCitation(null)}
      />
      <PdfCitationSheet
        citation={activePdfCitation}
        isOpen={!!activePdfCitation}
        onOpenChange={o => !o && setActivePdfCitation(null)}
      />
    </div>
  );
}
