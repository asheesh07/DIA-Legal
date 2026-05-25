import React from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { MessageSquare, FileText, AlertTriangle, UserMinus, ShieldAlert } from 'lucide-react';

const MODES = [
  { id: 'query', label: 'Query', icon: MessageSquare },
  { id: 'evidence', label: 'Evidence', icon: FileText },
  { id: 'contradictions', label: 'Contradictions', icon: AlertTriangle },
  { id: 'devils-advocate', label: "Devil's Advocate", icon: UserMinus },
  { id: 'brief', label: 'Brief', icon: ShieldAlert },
];

export function ModeTabs({ activeMode, onModeChange }) {
  return (
    <>
      <div className="hidden lg:flex">
        <Tabs value={activeMode} onValueChange={onModeChange} className="w-full">
          <TabsList className="bg-transparent border-b border-border rounded-none h-14 p-0">
            {MODES.map(mode => (
              <TabsTrigger 
                key={mode.id} 
                value={mode.id}
                className="data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:text-foreground data-[state=active]:shadow-none rounded-none border-b-2 border-transparent px-4 py-4 text-muted-foreground transition-none hover:text-foreground"
              >
                <mode.icon className="w-4 h-4 mr-2" />
                {mode.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex lg:hidden flex-1 max-w-[200px]">
        <Select value={activeMode} onValueChange={onModeChange}>
          <SelectTrigger className="h-10">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODES.map(mode => (
              <SelectItem key={mode.id} value={mode.id}>
                <div className="flex items-center">
                  <mode.icon className="w-4 h-4 mr-2" />
                  {mode.label}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </>
  );
}
