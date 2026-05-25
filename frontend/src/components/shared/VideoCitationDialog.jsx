import React, { useRef, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { PlayCircle, Users, Clock } from 'lucide-react';

function fmtTime(sec) {
  if (sec == null) return '?';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

export function VideoCitationDialog({ citation, isOpen, onOpenChange }) {
  const videoRef = useRef(null);

  const startSec  = citation?.time_range?.[0] ?? 0;
  const endSec    = citation?.time_range?.[1] ?? 0;
  const speakers  = (citation?.speakers || []).filter(Boolean);
  const timeLabel = endSec > 0 ? `${fmtTime(startSec)} – ${fmtTime(endSec)}` : null;

  useEffect(() => {
    if (!isOpen || !videoRef.current || !citation?.url) return;
    videoRef.current.src = citation.url;
    if (startSec > 0) {
      const onMeta = () => {
        videoRef.current.currentTime = startSec;
        videoRef.current.play().catch(() => {});
      };
      videoRef.current.addEventListener('loadedmetadata', onMeta, { once: true });
    }
  }, [isOpen, citation, startSec]);

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-5 pb-3">
          <DialogTitle className="flex items-center gap-2">
            <PlayCircle className="w-4 h-4 text-primary" />
            {citation?.citation_ref || citation?.ref || 'Video Source'}
          </DialogTitle>
          <div className="flex flex-wrap gap-2 mt-2">
            {timeLabel && (
              <Badge variant="secondary" className="gap-1 text-xs">
                <Clock className="w-3 h-3" />
                {timeLabel}
              </Badge>
            )}
            {speakers.map((sp, i) => (
              <Badge key={i} variant="outline" className="gap-1 text-xs">
                <Users className="w-3 h-3" />
                {sp.replace(/^SPEAKER_0*/i, 'Speaker ')}
              </Badge>
            ))}
          </div>
        </DialogHeader>
        <div className="px-6 pb-6">
          {citation?.url ? (
            <video
              ref={videoRef}
              controls
              className="w-full rounded-lg bg-black"
              style={{ maxHeight: '460px' }}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-36 text-muted-foreground text-sm rounded-lg bg-muted gap-2">
              <PlayCircle className="w-8 h-8 opacity-30" />
              <span>Video file not available for direct playback.</span>
              {timeLabel && <span className="text-xs">Timestamp: {timeLabel}</span>}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
