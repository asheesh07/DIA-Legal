import React, { useState, useRef, useEffect } from 'react';
import { query, getChatSessions, getChatHistory, saveChatHistory, deleteChatSession } from '@/api';
import { ModeFrame } from '@/components/shared/ModeFrame';
import { EmptyState } from '@/components/shared/EmptyState';
import { CitationBadge } from '@/components/shared/CitationBadge';
import { VideoCitationDialog } from '@/components/shared/VideoCitationDialog';
import { PdfCitationSheet } from '@/components/shared/PdfCitationSheet';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { History, Send, MessageSquare, Trash, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

const SUGGESTIONS = [
  'Summarise this case in 3 sentences',
  'What contradictions exist in the testimony?',
  'Who was present at the scene?'
];

export default function QueryMode({ caseId }) {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  // Citation viewer states
  const [activeVideoCitation, setActiveVideoCitation] = useState(null);
  const [activePdfCitation, setActivePdfCitation] = useState(null);
  
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!caseId) return;
    let isMounted = true;
    getChatSessions(caseId)
      .then(res => {
        if (isMounted && res.data?.sessions) {
          setSessions(res.data.sessions);
          if (res.data.sessions.length > 0) {
            setActiveSessionId(res.data.sessions[0].session_id);
          } else {
            setActiveSessionId(null);
          }
        }
      })
      .catch(e => console.error("Failed to load chat sessions:", e));
    
    return () => { isMounted = false; setSessions([]); setActiveSessionId(null); setMessages([]); };
  }, [caseId]);

  useEffect(() => {
    if (!caseId) return;
    if (!activeSessionId) {
      setMessages([]);
      return;
    }
    let isMounted = true;
    getChatHistory(caseId, activeSessionId)
      .then(res => {
        if (isMounted && res.data?.messages) {
          setMessages(res.data.messages);
        }
      })
      .catch(e => console.error("Failed to load chat history:", e));
    return () => { isMounted = false; };
  }, [caseId, activeSessionId]);

  const startNewChat = () => {
    setActiveSessionId(null);
    setMessages([]);
    setHistoryOpen(false);
  };

  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await deleteChatSession(caseId, sessionId);
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      toast.success("Chat deleted");
    } catch (err) {
      toast.error("Failed to delete chat");
    }
  };

  const send = async (text) => {
    if (loading) return;
    const q = (text || input).trim();
    if (!q || !caseId) {
      if (!caseId) toast.error("No active case selected");
      return;
    }

    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      currentSessionId = `chat_${Date.now()}`;
      setActiveSessionId(currentSessionId);
      setSessions(prev => [{ session_id: currentSessionId, title: q.slice(0, 30) }, ...prev]);
    }

    const title = q.slice(0, 30);

    setMessages(prev => {
      const newMsgs = [...prev, { role: 'user', content: q }];
      saveChatHistory(caseId, currentSessionId, newMsgs, title).catch(console.error);
      return newMsgs;
    });
    setInput('');
    setLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const res = await query(caseId, q, 'evidence', history);
      const d = res.data;
      setMessages(prev => {
        const newMsgs = [...prev, {
          role: 'assistant',
          content: d.answer,
          citations: d.citations,
          confidence: d.confidence,
        }];
        saveChatHistory(caseId, currentSessionId, newMsgs, title).catch(console.error);
        return newMsgs;
      });
    } catch (e) {
      setMessages(prev => {
        const newMsgs = [...prev, {
          role: 'assistant',
          content: `Error: ${e.response?.data?.detail || e.message}`,
          error: true,
        }];
        saveChatHistory(caseId, currentSessionId, newMsgs, title).catch(console.error);
        return newMsgs;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleCitationClick = (citation) => {
    const hasTime = Array.isArray(citation.time_range) && citation.time_range[1] > 0;
    const hasPage = citation.page_start > 0 || citation.page_end > 0;
    const type = citation.type || (hasTime ? 'video' : hasPage ? 'pdf' : 'transcript');
    if (type === 'video') {
      setActiveVideoCitation(citation);
    } else if (type === 'pdf') {
      setActivePdfCitation(citation);
    }
  };

  return (
    <ModeFrame
      title="Query"
      subtitle={caseId ? "Ask questions about the evidence" : "No case selected"}
      actions={
        <Sheet open={historyOpen} onOpenChange={setHistoryOpen}>
          <SheetTrigger asChild>
            <Button variant="outline" size="sm">
              <History className="w-4 h-4 mr-2" />
              History
            </Button>
          </SheetTrigger>
          <SheetContent className="w-[300px] sm:w-[400px]">
            <SheetHeader className="mb-4">
              <SheetTitle>Chat History</SheetTitle>
            </SheetHeader>
            <Button onClick={startNewChat} className="w-full mb-4">
              <Plus className="w-4 h-4 mr-2" />
              New Chat
            </Button>
            <div className="space-y-2">
              {sessions.map(s => (
                <div 
                  key={s.session_id}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors group",
                    activeSessionId === s.session_id ? "bg-primary text-primary-foreground" : "hover:bg-muted"
                  )}
                  onClick={() => { setActiveSessionId(s.session_id); setHistoryOpen(false); }}
                >
                  <span className="truncate text-sm font-medium">{s.title || "New Chat"}</span>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className={cn("w-6 h-6 opacity-0 group-hover:opacity-100", activeSessionId === s.session_id ? "text-primary-foreground hover:bg-primary-foreground/20" : "text-muted-foreground")}
                    onClick={(e) => handleDeleteSession(s.session_id, e)}
                  >
                    <Trash className="w-4 h-4" />
                  </Button>
                </div>
              ))}
              {sessions.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">No chat history.</p>
              )}
            </div>
          </SheetContent>
        </Sheet>
      }
    >
      <div className="flex flex-col h-full relative max-w-3xl mx-auto">
        <div className="flex-1 overflow-y-auto pb-32">
          {messages.length === 0 ? (
            <EmptyState
              icon={MessageSquare}
              title="How can I help with the case?"
              description="Query testimony, cross-reference documents, and find contradictions."
              suggestions={SUGGESTIONS}
              onSuggestionClick={send}
            />
          ) : (
            <div className="space-y-6 py-4">
              {messages.map((msg, i) => (
                <div key={i} className={cn("flex w-full", msg.role === 'user' ? "justify-end" : "justify-start")}>
                  <div className={cn(
                    "max-w-[85%] rounded-2xl px-5 py-3.5",
                    msg.role === 'user' 
                      ? "bg-primary text-primary-foreground rounded-br-none" 
                      : "bg-card border border-border text-card-foreground rounded-bl-none shadow-sm"
                  )}>
                    <div className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                    
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-2 border-t border-border/10 pt-3">
                        {msg.citations.map((c, idx) => (
                          <CitationBadge key={idx} citation={c} onClick={handleCitationClick} />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start w-full">
                  <div className="bg-card border border-border text-card-foreground rounded-2xl rounded-bl-none px-5 py-3.5 shadow-sm">
                    <div className="flex space-x-1 items-center h-5">
                      <div className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="absolute bottom-0 left-0 right-0 bg-background pt-4 pb-safe">
          <div className="flex items-end gap-2 p-2 border border-border rounded-xl bg-card shadow-sm focus-within:ring-1 focus-within:ring-primary transition-shadow">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask a question... (Shift+Enter for new line)"
              className="min-h-[40px] max-h-[200px] border-0 shadow-none focus-visible:ring-0 resize-none py-2 px-3 text-sm flex-1"
              rows={1}
            />
            <Button 
              size="icon" 
              className="mb-0.5 shrink-0 h-9 w-9 rounded-lg"
              disabled={!input.trim() || loading || !caseId}
              onClick={() => send()}
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
      <VideoCitationDialog citation={activeVideoCitation} isOpen={!!activeVideoCitation} onOpenChange={(o) => !o && setActiveVideoCitation(null)} />
      <PdfCitationSheet citation={activePdfCitation} isOpen={!!activePdfCitation} onOpenChange={(o) => !o && setActivePdfCitation(null)} />
    </ModeFrame>
  );
}
