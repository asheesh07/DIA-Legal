import React from 'react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';

export function EmptyState({ icon: Icon, title, description, action, suggestions = [], onSuggestionClick, className }) {
  return (
    <div className={cn("flex flex-col items-center justify-center h-full text-center p-8 space-y-6 max-w-md mx-auto", className)}>
      <div className="p-4 bg-muted rounded-full">
        <Icon className="w-8 h-8 text-muted-foreground" />
      </div>
      <div className="space-y-2">
        <h3 className="text-xl font-medium tracking-tight">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {action && (
        <div className="pt-2">
          {action}
        </div>
      )}
      {suggestions.length > 0 && (
        <div className="pt-6 w-full">
          <p className="text-xs font-medium text-muted-foreground mb-3 uppercase tracking-wider">Suggestions</p>
          <div className="flex flex-wrap justify-center gap-2">
            {suggestions.map((suggestion, idx) => (
              <Badge 
                key={idx} 
                variant="secondary" 
                className="cursor-pointer hover:bg-secondary/80 transition-colors"
                onClick={() => onSuggestionClick?.(suggestion)}
              >
                {suggestion}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
