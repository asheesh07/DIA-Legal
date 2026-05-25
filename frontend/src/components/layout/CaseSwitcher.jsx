import React, { useEffect, useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import * as api from '@/api';

export function CaseSwitcher({ activeCaseId, onCaseChange, cases = [] }) {
  return (
    <div className="w-[200px]">
      <Select value={activeCaseId || ''} onValueChange={onCaseChange}>
        <SelectTrigger className="h-10 bg-background border-none shadow-none font-medium">
          <SelectValue placeholder="Select a case" />
        </SelectTrigger>
        <SelectContent>
          {cases.length === 0 ? (
            <SelectItem value="empty" disabled>No cases found</SelectItem>
          ) : (
            cases.map(c => {
              const id   = (typeof c === 'string') ? c : (c.case_id || c.id || c.name || '');
              const name = (typeof c === 'string') ? c : (c.name || c.case_id || id);
              return (
                <SelectItem key={id} value={id}>
                  {name}
                </SelectItem>
              );
            })
          )}
        </SelectContent>
      </Select>
    </div>
  );
}
