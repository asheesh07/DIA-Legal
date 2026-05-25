import React from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

export function SkeletonCard({ rows = 3, height = "h-4", className }) {
  return (
    <div className={cn("flex flex-col space-y-3 p-4 border border-border rounded-xl", className)}>
      <Skeleton className="h-[125px] w-full rounded-xl" />
      <div className="space-y-2">
        {Array.from({ length: rows }).map((_, i) => (
          <Skeleton key={i} className={cn(height, "w-full", i === rows - 1 ? "w-[80%]" : "")} />
        ))}
      </div>
    </div>
  );
}
