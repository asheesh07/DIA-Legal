import React from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Plus, Trash2, MoreVertical } from 'lucide-react';
import {
  DropdownMenu, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

export function SessionSidebar({ cases, activeCaseId, onCaseSelect, onCreateNew, onDeleteCase }) {
  return (
    <div className="w-[210px] shrink-0 flex flex-col h-full bg-slate-950 text-slate-300 border-r border-slate-800">
      {/* Brand */}
      <div className="px-4 pt-5 pb-4">
        <span className="font-bold text-white text-base tracking-tight">DIA Legal</span>
      </div>

      {/* New session */}
      <div className="px-3 mb-4">
        <Button
          onClick={onCreateNew}
          size="sm"
          className="w-full"
        >
          <Plus className="w-4 h-4 mr-2" />
          New session
        </Button>
      </div>

      {/* Label */}
      <p className="px-4 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
        Sessions
      </p>

      {/* List */}
      <ScrollArea className="flex-1 px-2">
        <div className="space-y-0.5 pb-4">
          {cases.length === 0 && (
            <p className="text-xs text-slate-500 px-3 py-3 leading-relaxed">
              No sessions yet — create one to get started.
            </p>
          )}
          {cases.map(c => {
            const id   = typeof c === 'string' ? c : (c.case_id || c.id || c.name || String(c));
            const name = typeof c === 'string' ? c : (c.name || c.case_id || id);
            const isActive = id === activeCaseId;
            return (
              <div
                key={id}
                className={cn(
                  'group flex items-center justify-between px-3 py-2 rounded-md cursor-pointer transition-colors text-sm',
                  isActive
                    ? 'bg-primary/15 text-primary-foreground border border-primary/25'
                    : 'text-slate-300 hover:bg-slate-800/70'
                )}
                onClick={() => onCaseSelect(id)}
              >
                <span className="truncate flex-1 leading-snug">{name}</span>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild onClick={e => e.stopPropagation()}>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="w-5 h-5 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-white shrink-0 ml-1"
                    >
                      <MoreVertical className="w-3 h-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      className="text-destructive focus:bg-destructive/10"
                      onClick={e => { e.stopPropagation(); onDeleteCase?.(id); }}
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete session
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
