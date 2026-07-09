import React, { useState, lazy, Suspense } from 'react';
import {
  Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription,
} from '@/components/ui/sheet';
import { Gavel } from 'lucide-react';

const BriefMode = lazy(() => import('@/components/modes/BriefMode'));

function Spinner() {
  return (
    <div className="flex items-center justify-center h-40">
      <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

export function PrepareForTrialSheet({ caseId, open, onOpenChange }) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full sm:w-[780px] sm:max-w-[90vw] overflow-y-auto p-0"
      >
        <SheetHeader className="px-6 pt-6 pb-4 border-b border-border">
          <SheetTitle className="flex items-center gap-2 text-base">
            <Gavel className="w-4 h-4 text-primary" />
            Prepare for Trial
          </SheetTitle>
          <SheetDescription className="text-xs text-muted-foreground">
            Enter your legal position and generate a full pre-trial brief from your case files.
          </SheetDescription>
        </SheetHeader>

        <div className="px-6 py-4">
          <Suspense fallback={<Spinner />}>
            <BriefMode caseId={caseId} embedded />
          </Suspense>
        </div>
      </SheetContent>
    </Sheet>
  );
}
