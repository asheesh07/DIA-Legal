import React from 'react';
import { Badge } from '@/components/ui/badge';

export function SeverityBadge({ severity, className }) {
  const norm = (severity || "").toLowerCase();
  
  if (norm === "high") {
    return <Badge variant="destructive" className={className}>High</Badge>;
  }
  if (norm === "medium") {
    return <Badge className={`bg-warning hover:bg-warning/80 text-warning-foreground ${className || ""}`}>Medium</Badge>;
  }
  if (norm === "low") {
    return <Badge className={`bg-success hover:bg-success/80 text-success-foreground ${className || ""}`}>Low</Badge>;
  }
  
  return <Badge variant="secondary" className={className}>{severity}</Badge>;
}
