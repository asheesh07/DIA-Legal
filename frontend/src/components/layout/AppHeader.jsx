import React from 'react';
import { Scale, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CaseSwitcher } from './CaseSwitcher';
import { ModeTabs } from './ModeTabs';
import { cn } from '@/lib/utils';

export function AppHeader({ cases, activeCaseId, onCaseChange, activeMode, onModeChange, onMobileMenuClick, className }) {
  return (
    <header className={cn("flex items-center h-14 border-b border-border bg-background px-4 lg:px-6 shrink-0", className)}>
      {/* Mobile menu */}
      <Button variant="ghost" size="icon" className="lg:hidden mr-2 -ml-2" onClick={onMobileMenuClick}>
        <Menu className="w-5 h-5" />
      </Button>

      {/* Logo / wordmark */}
      <div className="flex items-center gap-2 mr-6 shrink-0">
        <div className="w-7 h-7 bg-primary rounded-md flex items-center justify-center shadow-sm">
          <Scale className="w-4 h-4 text-primary-foreground" />
        </div>
        <span className="font-bold text-base tracking-tight hidden sm:inline">DIA-Legal</span>
        <Badge variant="secondary" className="text-[10px] px-1.5 py-0 hidden md:inline-flex">v2.0</Badge>
      </div>

      {/* Case switcher */}
      {cases.length > 0 && (
        <div className="hidden md:flex items-center gap-2 mr-6 text-sm text-muted-foreground shrink-0">
          <span className="text-xs">Case:</span>
          <CaseSwitcher cases={cases} activeCaseId={activeCaseId} onCaseChange={onCaseChange} />
        </div>
      )}

      {/* Mode tabs — centered */}
      <div className="flex-1 flex justify-center lg:justify-start min-w-0">
        <ModeTabs activeMode={activeMode} onModeChange={onModeChange} />
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1 ml-auto pl-4 shrink-0">
        <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground text-xs gap-1.5" asChild>
          <a href="https://github.com/asheesh07/DIA-Legal" target="_blank" rel="noreferrer">
            {/* GitHub mark SVG — lucide-react v1 removed Github icon */}
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
            </svg>
            GitHub
          </a>
        </Button>
      </div>
    </header>
  );
}
