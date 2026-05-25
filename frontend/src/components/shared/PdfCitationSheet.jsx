import React from 'react';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { FileText, BookOpen } from 'lucide-react';

export function PdfCitationSheet({ citation, isOpen, onOpenChange }) {
  const pageStart    = citation?.page_start || 0;
  const pageEnd      = citation?.page_end   || pageStart;
  const sectionTitle = citation?.section_title || '';
  const originalName = citation?.original_name || '';
  const pageLabel    = pageStart > 0
    ? (pageStart === pageEnd ? `Page ${pageStart}` : `Pages ${pageStart}–${pageEnd}`)
    : null;

  const src = citation?.url
    ? `${citation.url}${pageStart ? `#page=${pageStart}` : ''}`
    : null;

  return (
    <Sheet open={isOpen} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[90vw] sm:w-[640px] p-0 flex flex-col">
        <SheetHeader className="px-6 py-4 border-b border-border shrink-0">
          <SheetTitle className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" />
            {originalName || citation?.citation_ref || citation?.ref || 'PDF Source'}
          </SheetTitle>
          <div className="flex flex-wrap gap-2 mt-1.5">
            {pageLabel && (
              <Badge variant="secondary" className="gap-1 text-xs">
                <BookOpen className="w-3 h-3" />
                {pageLabel}
              </Badge>
            )}
            {sectionTitle && (
              <Badge variant="outline" className="text-xs max-w-[280px] truncate">
                {sectionTitle}
              </Badge>
            )}
          </div>
        </SheetHeader>
        <div className="flex-1 min-h-0">
          {src ? (
            <iframe
              src={src}
              className="w-full h-full border-0"
              title={originalName || 'PDF'}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm gap-3 px-6">
              <FileText className="w-10 h-10 opacity-20" />
              <p>PDF file not available for direct preview.</p>
              {(sectionTitle || pageLabel) && (
                <div className="text-center space-y-1">
                  {sectionTitle && <p className="text-xs font-medium">{sectionTitle}</p>}
                  {pageLabel && <p className="text-xs">{pageLabel}</p>}
                </div>
              )}
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
