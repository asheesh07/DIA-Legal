import React, { useState, useRef } from 'react';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { ingestYoutubeStream, ingestVideoStream, ingestPDFsStream } from '@/api';
import {
  Plus, Trash2, Link2, Video, FileText,
  Loader2, CheckCircle2, XCircle, Upload,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ── Limits ───────────────────────────────────────────────────────
const PDF_MAX_FILES   = 5;
const PDF_MAX_MB      = 12;
const VIDEO_MAX_FILES = 2;
const VIDEO_MAX_MB    = 250;

const PDF_MAX_BYTES   = PDF_MAX_MB   * 1024 * 1024;
const VIDEO_MAX_BYTES = VIDEO_MAX_MB * 1024 * 1024;

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

// ── Stage metadata ────────────────────────────────────────────────
const STAGE_LABEL = {
  uploading:        'Uploading…',
  fetching_stream:  'Fetching stream…',
  extracting_audio: 'Extracting audio…',
  sampling_frames:  'Sampling frames…',
  transcribing:     'Transcribing…',
  diarizing:        'Identifying speakers…',
  chunking:         'Chunking…',
  embedding:        'Embedding…',
  indexing:         'Indexing…',
  reading:          'Reading PDF…',
  file_done:        'Done',
  cached:           'Already indexed',
  done:             'Done',
  error:            'Failed',
};

const VIDEO_STAGES  = ['uploading','extracting_audio','sampling_frames','transcribing','diarizing','chunking','embedding','indexing'];
const YT_STAGES     = ['fetching_stream','extracting_audio','sampling_frames','transcribing','diarizing','chunking','embedding','indexing'];
const PDF_STAGES    = ['reading','chunking','embedding','indexing'];

function stageProgress(stage, stageList) {
  const idx = stageList.indexOf(stage);
  if (idx < 0) return stage === 'done' || stage === 'file_done' || stage === 'cached' ? 100 : 0;
  return Math.round(((idx + 1) / stageList.length) * 100);
}

// ── Sub-components ────────────────────────────────────────────────
function SectionHeader({ icon: Icon, iconClass, title }) {
  return (
    <div className="flex items-center gap-2">
      <Icon className={cn('w-4 h-4', iconClass)} />
      <span className="text-sm font-medium">{title}</span>
    </div>
  );
}

function StatusIcon({ status }) {
  if (!status || status === 'pending') return <span className="w-4 h-4" />;
  if (status === 'running')  return <Loader2 className="w-4 h-4 animate-spin text-primary shrink-0" />;
  if (status === 'success' || status === 'cached') return <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />;
  return <XCircle className="w-4 h-4 text-destructive shrink-0" />;
}

function StageBar({ stage, stageList, status }) {
  if (!stage || status === 'pending') return null;
  const pct = stageProgress(stage, stageList);
  const isDone = status === 'success' || status === 'cached';
  const isErr  = status === 'error';
  return (
    <div className="mt-1 space-y-0.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-muted-foreground">
          {STAGE_LABEL[stage] ?? stage}
        </span>
        <span className="text-[11px] text-muted-foreground">{pct}%</span>
      </div>
      <div className="h-1 rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500',
            isDone ? 'bg-green-500' : isErr ? 'bg-destructive' : 'bg-primary'
          )}
          style={{ width: `${isDone ? 100 : pct}%` }}
        />
      </div>
    </div>
  );
}

// ── Main dialog ───────────────────────────────────────────────────
export function IngestDialog({ isOpen, onOpenChange, activeCaseId, onIngested }) {
  const [ytUrls,     setYtUrls]     = useState(['']);
  const [videoFiles, setVideoFiles] = useState([]);
  const [pdfFiles,   setPdfFiles]   = useState([]);
  const [items,      setItems]      = useState({});   // key → { status, stage, error, chunks }
  const [isRunning,  setIsRunning]  = useState(false);
  const videoRef = useRef(null);
  const pdfRef   = useRef(null);

  const validYtUrls = ytUrls.filter(u => u.trim());
  const totalQueued = validYtUrls.length + videoFiles.length + pdfFiles.length;
  const hasStarted  = Object.keys(items).length > 0;
  const allDone     = hasStarted && Object.values(items).every(
    s => s.status === 'success' || s.status === 'cached' || s.status === 'error'
  );

  const setItem = (key, patch) =>
    setItems(prev => ({ ...prev, [key]: { ...(prev[key] || {}), ...patch } }));

  const reset = () => {
    setYtUrls(['']);
    setVideoFiles([]);
    setPdfFiles([]);
    setItems({});
    setIsRunning(false);
  };

  const handleClose = () => {
    if (!isRunning) { reset(); onOpenChange(false); }
  };

  const ingestAll = async () => {
    if (!activeCaseId) { toast.error('No active case selected'); return; }
    if (totalQueued === 0) { toast.error('No sources queued'); return; }

    setIsRunning(true);

    // Initialise status entries
    const init = {};
    validYtUrls.forEach((_, i) => { init[`yt-${i}`] = { status: 'pending', stage: null }; });
    videoFiles.forEach((_, i)  => { init[`vid-${i}`] = { status: 'pending', stage: null }; });
    if (pdfFiles.length) init['pdfs'] = { status: 'pending', stage: null };
    setItems(init);

    let anySuccess = false;

    // YouTube URLs — sequential to avoid saturating bandwidth
    for (let i = 0; i < validYtUrls.length; i++) {
      const key = `yt-${i}`;
      setItem(key, { status: 'running', stage: 'fetching_stream' });
      try {
        await ingestYoutubeStream(activeCaseId, validYtUrls[i], (ev) => {
          if (ev.stage === 'done') {
            setItem(key, { status: 'success', stage: 'done', chunks: ev.chunks });
            anySuccess = true;
          } else if (ev.stage === 'cached') {
            setItem(key, { status: 'cached', stage: 'cached', chunks: ev.chunks });
            anySuccess = true;
          } else if (ev.stage === 'error') {
            setItem(key, { status: 'error', stage: 'error', error: ev.error });
          } else {
            setItem(key, { stage: ev.stage });
          }
        });
      } catch (e) {
        setItem(key, { status: 'error', stage: 'error', error: e.message });
      }
    }

    // Video files — one at a time (each is long-running)
    for (let i = 0; i < videoFiles.length; i++) {
      const key = `vid-${i}`;
      setItem(key, { status: 'running', stage: 'uploading' });
      try {
        await ingestVideoStream(activeCaseId, videoFiles[i], (ev) => {
          if (ev.stage === 'done') {
            setItem(key, { status: 'success', stage: 'done', chunks: ev.chunks });
            anySuccess = true;
          } else if (ev.stage === 'error') {
            setItem(key, { status: 'error', stage: 'error', error: ev.error });
          } else {
            setItem(key, { stage: ev.stage });
          }
        });
      } catch (e) {
        setItem(key, { status: 'error', stage: 'error', error: e.message });
      }
    }

    // PDFs — batched together
    if (pdfFiles.length > 0) {
      setItem('pdfs', { status: 'running', stage: 'reading' });
      try {
        await ingestPDFsStream(activeCaseId, pdfFiles, (ev) => {
          if (ev.stage === 'done') {
            setItem('pdfs', { status: 'success', stage: 'done', results: ev.results });
            anySuccess = true;
          } else if (ev.stage === 'file_error') {
            // individual file failed — keep running
            setItem('pdfs', { stage: 'reading' });
          } else if (ev.stage === 'error') {
            setItem('pdfs', { status: 'error', stage: 'error', error: ev.error });
          } else {
            setItem('pdfs', { stage: ev.stage });
          }
        });
      } catch (e) {
        setItem('pdfs', { status: 'error', stage: 'error', error: e.message });
      }
    }

    setIsRunning(false);
    if (anySuccess) {
      onIngested?.();
      toast.success('Sources ingested successfully');
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[520px] max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add Sources</DialogTitle>
          {activeCaseId && (
            <p className="text-sm text-muted-foreground">
              Case: <span className="font-medium text-foreground">{activeCaseId}</span>
            </p>
          )}
        </DialogHeader>

        <div className="space-y-5 py-2">

          {/* ── YouTube ─────────────────────────────────────────── */}
          <div className="space-y-2">
            <SectionHeader icon={Link2} iconClass="text-red-500" title="YouTube URLs" />
            <div className="space-y-2">
              {ytUrls.map((url, i) => {
                const it  = items[`yt-${i}`] || {};
                const key = `yt-${i}`;
                return (
                  <div key={i} className="space-y-0">
                    <div className="flex gap-2 items-center">
                      <Input
                        value={url}
                        onChange={e => setYtUrls(prev => prev.map((u, idx) => idx === i ? e.target.value : u))}
                        placeholder="https://youtube.com/watch?v=…"
                        disabled={isRunning}
                        className="flex-1 text-sm"
                      />
                      <div className="w-7 flex items-center justify-center shrink-0">
                        {it.status ? (
                          <StatusIcon status={it.status} />
                        ) : ytUrls.length > 1 && !isRunning ? (
                          <button
                            onClick={() => setYtUrls(prev => prev.filter((_, idx) => idx !== i))}
                            className="text-muted-foreground hover:text-destructive transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        ) : null}
                      </div>
                    </div>
                    {it.status === 'running' && (
                      <StageBar stage={it.stage} stageList={YT_STAGES} status={it.status} />
                    )}
                    {it.status === 'error' && (
                      <p className="text-[11px] text-destructive mt-0.5">{it.error}</p>
                    )}
                  </div>
                );
              })}
              {!isRunning && (
                <Button
                  variant="ghost" size="sm"
                  className="text-muted-foreground hover:text-primary pl-0"
                  onClick={() => setYtUrls(prev => [...prev, ''])}
                >
                  <Plus className="w-3.5 h-3.5 mr-1.5" />
                  Add URL
                </Button>
              )}
            </div>
          </div>

          <Separator />

          {/* ── Video files ──────────────────────────────────────── */}
          <div className="space-y-2">
            <SectionHeader icon={Video} iconClass="text-blue-500"
              title={`Video Files (${videoFiles.length}/${VIDEO_MAX_FILES} · max ${VIDEO_MAX_MB} MB each)`} />
            <input ref={videoRef} type="file" accept=".mp4,.mov,.mkv,.avi,.webm"
              multiple className="hidden" disabled={isRunning}
              onChange={e => {
                const picked = Array.from(e.target.files);
                e.target.value = '';
                setVideoFiles(prev => {
                  const tooBig = picked.filter(f => f.size > VIDEO_MAX_BYTES);
                  const ok     = picked.filter(f => f.size <= VIDEO_MAX_BYTES);
                  tooBig.forEach(f =>
                    toast.error(
                      `"${f.name}" is ${fmtSize(f.size)} — video files must be ${VIDEO_MAX_MB} MB or smaller`
                    )
                  );
                  const combined = [...prev, ...ok];
                  if (combined.length > VIDEO_MAX_FILES) {
                    const removed = combined.length - VIDEO_MAX_FILES;
                    toast.error(
                      `You added ${combined.length} video${combined.length !== 1 ? 's' : ''} — the limit is ${VIDEO_MAX_FILES}. ` +
                      `${removed} file${removed !== 1 ? 's were' : ' was'} removed.`
                    );
                    return combined.slice(0, VIDEO_MAX_FILES);
                  }
                  return combined;
                });
              }}
            />
            {videoFiles.length > 0 && (
              <div className="border rounded-md px-3 divide-y divide-border">
                {videoFiles.map((f, i) => {
                  const it = items[`vid-${i}`] || {};
                  return (
                    <div key={i} className="py-1.5 space-y-0">
                      <div className="flex items-center gap-2">
                        <Video className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                        <span className="text-sm text-muted-foreground flex-1 truncate">{f.name}</span>
                        {it.status ? (
                          <StatusIcon status={it.status} />
                        ) : !isRunning ? (
                          <button
                            onClick={() => setVideoFiles(prev => prev.filter((_, idx) => idx !== i))}
                            className="text-muted-foreground hover:text-destructive transition-colors shrink-0"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        ) : null}
                      </div>
                      {it.status === 'running' && (
                        <StageBar stage={it.stage} stageList={VIDEO_STAGES} status={it.status} />
                      )}
                      {it.status === 'error' && (
                        <p className="text-[11px] text-destructive mt-0.5">{it.error}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
            <Button variant="outline" size="sm" className="w-full"
              onClick={() => videoRef.current?.click()}
              disabled={isRunning || videoFiles.length >= VIDEO_MAX_FILES}>
              <Upload className="w-3.5 h-3.5 mr-2" />
              {videoFiles.length >= VIDEO_MAX_FILES ? `Limit reached (${VIDEO_MAX_FILES}/${VIDEO_MAX_FILES})` : 'Choose video files…'}
            </Button>
          </div>

          <Separator />

          {/* ── PDF files ────────────────────────────────────────── */}
          <div className="space-y-2">
            <SectionHeader icon={FileText} iconClass="text-orange-500"
              title={`PDF Documents (${pdfFiles.length}/${PDF_MAX_FILES} · max ${PDF_MAX_MB} MB each)`} />
            <input ref={pdfRef} type="file" accept=".pdf" multiple className="hidden"
              disabled={isRunning}
              onChange={e => {
                const picked = Array.from(e.target.files);
                e.target.value = '';
                setPdfFiles(prev => {
                  const tooBig = picked.filter(f => f.size > PDF_MAX_BYTES);
                  const ok     = picked.filter(f => f.size <= PDF_MAX_BYTES);
                  tooBig.forEach(f =>
                    toast.error(
                      `"${f.name}" is ${fmtSize(f.size)} — PDF files must be ${PDF_MAX_MB} MB or smaller`
                    )
                  );
                  const combined = [...prev, ...ok];
                  if (combined.length > PDF_MAX_FILES) {
                    const removed = combined.length - PDF_MAX_FILES;
                    toast.error(
                      `You added ${combined.length} PDF${combined.length !== 1 ? 's' : ''} — the limit is ${PDF_MAX_FILES}. ` +
                      `${removed} file${removed !== 1 ? 's were' : ' was'} removed.`
                    );
                    return combined.slice(0, PDF_MAX_FILES);
                  }
                  return combined;
                });
              }}
            />
            {pdfFiles.length > 0 && (
              <div className="border rounded-md px-3 divide-y divide-border">
                {pdfFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 py-1.5">
                    <FileText className="w-3.5 h-3.5 text-orange-400 shrink-0" />
                    <span className="text-sm text-muted-foreground flex-1 truncate">{f.name}</span>
                    {!isRunning && !items['pdfs'] && (
                      <button
                        onClick={() => setPdfFiles(prev => prev.filter((_, idx) => idx !== i))}
                        className="text-muted-foreground hover:text-destructive transition-colors shrink-0"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))}

                {/* PDF batch progress */}
                {(() => {
                  const it = items['pdfs'] || {};
                  if (!it.status) return null;
                  return (
                    <div className="py-1.5 space-y-0">
                      <div className="flex items-center gap-2">
                        <StatusIcon status={it.status} />
                        <span className="text-xs text-muted-foreground">
                          {it.status === 'running'
                            ? (STAGE_LABEL[it.stage] ?? 'Processing…')
                            : it.status === 'success'
                            ? 'All PDFs indexed'
                            : it.status === 'cached'
                            ? 'Already indexed'
                            : it.error ?? 'Failed'}
                        </span>
                      </div>
                      {it.status === 'running' && (
                        <StageBar stage={it.stage} stageList={PDF_STAGES} status={it.status} />
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
            <Button variant="outline" size="sm" className="w-full"
              onClick={() => pdfRef.current?.click()}
              disabled={isRunning || pdfFiles.length >= PDF_MAX_FILES}>
              <Upload className="w-3.5 h-3.5 mr-2" />
              {pdfFiles.length >= PDF_MAX_FILES ? `Limit reached (${PDF_MAX_FILES}/${PDF_MAX_FILES})` : 'Choose PDF files…'}
            </Button>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={handleClose} disabled={isRunning}>
            {allDone ? 'Close' : 'Cancel'}
          </Button>
          {!allDone && (
            <Button onClick={ingestAll} disabled={isRunning || totalQueued === 0}>
              {isRunning ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Ingesting…</>
              ) : totalQueued > 0 ? (
                `Ingest ${totalQueued} source${totalQueued > 1 ? 's' : ''}`
              ) : 'Ingest'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
