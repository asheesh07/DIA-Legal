import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SeverityBadge } from './SeverityBadge';
import { CitationBadge } from './CitationBadge';
import { ArrowLeftRight } from 'lucide-react';

function speakerLabel(speaker, sourceName) {
  if (!speaker) return sourceName || 'Source';
  if (speaker === 'DOCUMENT') return sourceName || 'Document';
  if (speaker === 'Q') return 'Q (Questioner)';
  if (speaker === 'A') return 'A (Witness)';
  if (/^W\d+$/.test(speaker)) return `Witness ${speaker.slice(1)}`;
  if (speaker.toUpperCase().startsWith('SPEAKER_'))
    return `Speaker ${speaker.replace(/^SPEAKER_0*/i, '')}`;
  return speaker;
}

function sourceHeader(sourceName, timestamp) {
  const parts = [sourceName, timestamp].filter(Boolean);
  return parts.length ? parts.join(' · ') : null;
}

export function ContradictionCard({ contradiction, onCitationClick }) {
  const isCross = contradiction.is_cross_source;
  const labelA = speakerLabel(contradiction.speaker_a, contradiction.source_name_a);
  const labelB = speakerLabel(contradiction.speaker_b, contradiction.source_name_b);

  return (
    <Card className="overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-muted/40 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <SeverityBadge severity={contradiction.severity} />
          {isCross && (
            <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5 text-xs">
              <ArrowLeftRight className="w-3 h-3 mr-1" />
              Cross-source
            </Badge>
          )}
        </div>
        {contradiction.explanation && (
          <p className="text-xs text-muted-foreground leading-snug max-w-prose text-right hidden sm:block">
            {contradiction.explanation}
          </p>
        )}
      </div>

      {/* Two-panel body */}
      <CardContent className="p-0">
        <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border">
          {/* Side A */}
          <div className="p-4 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {labelA}
              </span>
              {contradiction.claim_citation && (
                <CitationBadge citation={contradiction.claim_citation} onClick={onCitationClick} />
              )}
            </div>
            {sourceHeader(contradiction.source_name_a, contradiction.timestamp_a) && (
              <p className="text-[11px] font-mono text-muted-foreground leading-tight">
                {sourceHeader(contradiction.source_name_a, contradiction.timestamp_a)}
              </p>
            )}
            <blockquote className="text-sm leading-relaxed border-l-2 border-primary/40 pl-3 italic text-foreground/90">
              "{contradiction.claim}"
            </blockquote>
          </div>

          {/* Side B */}
          <div className="p-4 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {labelB}
              </span>
              {contradiction.evidence_citation && (
                <CitationBadge citation={contradiction.evidence_citation} onClick={onCitationClick} />
              )}
            </div>
            {sourceHeader(contradiction.source_name_b, contradiction.timestamp_b) && (
              <p className="text-[11px] font-mono text-muted-foreground leading-tight">
                {sourceHeader(contradiction.source_name_b, contradiction.timestamp_b)}
              </p>
            )}
            <blockquote className="text-sm leading-relaxed border-l-2 border-destructive/40 pl-3 italic text-foreground/90">
              "{contradiction.evidence}"
            </blockquote>
          </div>
        </div>

        {/* Explanation — shown below on mobile */}
        {contradiction.explanation && (
          <div className="sm:hidden px-4 pb-3 pt-1 border-t border-border">
            <p className="text-xs text-muted-foreground">{contradiction.explanation}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
