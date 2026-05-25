import React from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Folder, Plus, MoreVertical, Trash, Edit2, Upload } from 'lucide-react';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';

export function AppSidebar({ cases, activeCaseId, onCaseSelect, onCreateNew, onAddSource, onDeleteCase, collapsed, setCollapsed, className }) {
  return (
    <div className={cn("flex flex-col h-full bg-slate-950 text-slate-300 transition-all duration-300", collapsed ? "w-[68px]" : "w-[220px]", className)}>
      <div className="p-4 flex items-center justify-between">
        <span className={cn("font-semibold text-slate-100", collapsed && "hidden")}>Cases</span>
        <Button variant="ghost" size="icon" onClick={() => setCollapsed(!collapsed)} className="hidden lg:flex text-slate-400 hover:text-slate-100">
          <Folder className="w-5 h-5" />
        </Button>
      </div>
      <ScrollArea className="flex-1 px-2">
        <div className="space-y-1">
          {cases.map(c => {
            const id   = (typeof c === 'string') ? c : (c.case_id || c.id || c.name || String(c));
            const name = (typeof c === 'string') ? c : (c.name || c.case_id || id);
            const isActive = id === activeCaseId;
            return (
              <div 
                key={id} 
                className={cn(
                  "flex items-center justify-between px-3 py-2 rounded-md cursor-pointer transition-colors group",
                  isActive ? "bg-primary text-primary-foreground" : "hover:bg-slate-800"
                )}
                onClick={() => onCaseSelect(id)}
              >
                <div className="flex items-center truncate">
                  <div className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center text-xs font-medium mr-3 shrink-0">
                    {name.substring(0, 2).toUpperCase()}
                  </div>
                  {!collapsed && <span className="truncate">{name}</span>}
                </div>
                {!collapsed && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild onClick={e => e.stopPropagation()}>
                      <Button variant="ghost" size="icon" className="w-6 h-6 opacity-0 group-hover:opacity-100">
                        <MoreVertical className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={(e) => { e.stopPropagation(); /* handle rename */ }}>
                        <Edit2 className="w-4 h-4 mr-2" /> Rename
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-destructive focus:bg-destructive/10" onClick={(e) => { e.stopPropagation(); onDeleteCase?.(id); }}>
                        <Trash className="w-4 h-4 mr-2" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>
      <div className="p-3 mt-auto space-y-2">
        {activeCaseId && (
          <Button
            onClick={onAddSource}
            variant="secondary"
            className={cn("w-full bg-slate-800 text-slate-200 hover:bg-slate-700 border-0", collapsed && "px-0")}
          >
            <Upload className={cn("w-4 h-4", !collapsed && "mr-2")} />
            {!collapsed && "Add source"}
          </Button>
        )}
        <Button
          onClick={onCreateNew}
          variant="secondary"
          className={cn("w-full bg-slate-700 text-slate-300 hover:bg-slate-600 border-0", collapsed && "px-0")}
        >
          <Plus className={cn("w-4 h-4", !collapsed && "mr-2")} />
          {!collapsed && "New case"}
        </Button>
      </div>
    </div>
  );
}
