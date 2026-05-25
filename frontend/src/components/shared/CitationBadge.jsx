import React from 'react';
import { Badge } from '@/components/ui/badge';
import { FileText, PlayCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

function fmtHMS(sec) {
  if (sec == null) return null;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

export function CitationBadge({ citation, onClick, className }) {
  if (!citation) return null;

  const hasTimeRange = Array.isArray(citation.time_range) && citation.time_range[1] > 0;
  const hasPageInfo  = citation.page_start > 0 || citation.page_end > 0;

  let type = citation.type || 'unknown';
  if (type === 'unknown' || type === 'transcript') {
    if (hasTimeRange) type = 'video';
    else if (hasPageInfo) type = 'pdf';
  }

  let Icon  = FileText;
  let label = citation.citation_ref || citation.ref || 'Source';

  if (type === 'video') {
    Icon = PlayCircle;
    const t0 = fmtHMS(citation.time_range?.[0]);
    const firstSpeaker = (citation.speakers || []).find(Boolean);
    if (t0 && firstSpeaker) label = `${t0} · ${firstSpeaker}`;
    else if (t0) label = t0;
  } else if (type === 'pdf') {
    Icon = FileText;
    const exhibitId    = citation.exhibit_id;
    const exhibitTitle = citation.exhibit_title;
    if (exhibitId || exhibitTitle) {
      label = exhibitId && exhibitTitle
        ? `Exhibit ${exhibitId} — ${exhibitTitle}`.slice(0, 42)
        : exhibitId
          ? `Exhibit ${exhibitId}`
          : (exhibitTitle.length > 28 ? exhibitTitle.slice(0, 28) + '…' : exhibitTitle);
    } else if (citation.original_name) {
      const n = citation.original_name.replace(/\.pdf$/i, '');
      label = n.length > 28 ? n.slice(0, 28) + '…' : n;
    } else if (citation.section_title) {
      const t = citation.section_title;
      label = t.length > 28 ? t.slice(0, 28) + '…' : t;
    } else if (citation.page_start) {
      label = citation.page_start === citation.page_end
        ? `p.${citation.page_start}`
        : `pp.${citation.page_start}–${citation.page_end}`;
    }
  }

  return (
    <Badge
      variant="outline"
      className={cn(
        'cursor-pointer hover:bg-primary/10 hover:text-primary transition-colors text-primary border-primary/20 bg-primary/5 gap-1 items-center',
        className
      )}
      onClick={() => onClick?.(citation)}
    >
      <Icon className="w-3 h-3 shrink-0" />
      <span>{label}</span>
    </Badge>
  );
}
