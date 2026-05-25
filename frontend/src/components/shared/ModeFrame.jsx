import React from 'react';
import { cn } from '@/lib/utils';

export function ModeFrame({ title, subtitle, actions, children, className }) {
  return (
    <div className={cn("flex flex-col h-full", className)}>
      <header className="flex flex-col sm:flex-row sm:items-center justify-between py-6 px-6 border-b border-border bg-card text-card-foreground">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
        </div>
        {actions && (
          <div className="mt-4 sm:mt-0 flex items-center space-x-2">
            {actions}
          </div>
        )}
      </header>
      <main className="flex-1 overflow-auto p-6 bg-background">
        {children}
      </main>
    </div>
  );
}
