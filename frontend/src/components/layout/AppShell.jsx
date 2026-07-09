import React, { useState, useEffect, useCallback, lazy, Suspense } from 'react';
import { SessionSidebar } from './SessionSidebar';
import { Workspace } from '@/components/workspace/Workspace';
import * as api from '@/api';

const NewCaseDialog = lazy(() => import('../ingest/NewCaseDialog').then(m => ({ default: m.NewCaseDialog })));

export function AppShell() {
  const [cases,         setCases]         = useState([]);
  const [activeCaseId,  setActiveCaseId]  = useState('');
  const [fileCount,     setFileCount]     = useState(0);
  const [isNewCaseOpen, setIsNewCaseOpen] = useState(false);

  const fetchCases = useCallback(async () => {
    try {
      const resp = await api.getCases();
      setCases(resp.data?.cases || []);
    } catch (err) {
      console.error('Failed to fetch cases:', err);
    }
  }, []);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const refreshFileCount = useCallback((caseId) => {
    if (!caseId) { setFileCount(0); return; }
    api.getCaseStats(caseId)
      .then(res => setFileCount(res.data?.document_count || 0))
      .catch(() => setFileCount(0));
  }, []);

  useEffect(() => { refreshFileCount(activeCaseId); }, [activeCaseId, refreshFileCount]);

  const handleCaseCreated = async (newCaseName) => {
    try {
      await api.createCase(newCaseName);
    } catch (err) {
      console.error('Failed to register session:', err);
    }
    await fetchCases();
    setActiveCaseId(newCaseName);
  };

  const handleDeleteCase = async (id) => {
    try {
      await api.deleteCase(id);
      fetchCases();
      if (activeCaseId === id) { setActiveCaseId(''); setFileCount(0); }
    } catch (err) {
      console.error(err);
    }
  };

  const handleIngested = useCallback(() => {
    fetchCases();
    refreshFileCount(activeCaseId);
  }, [fetchCases, refreshFileCount, activeCaseId]);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <SessionSidebar
        cases={cases}
        activeCaseId={activeCaseId}
        onCaseSelect={setActiveCaseId}
        onCreateNew={() => setIsNewCaseOpen(true)}
        onDeleteCase={handleDeleteCase}
      />

      <div className="flex-1 min-w-0">
        <Workspace
          caseId={activeCaseId}
          fileCount={fileCount}
          onIngested={handleIngested}
          onCreateCase={() => setIsNewCaseOpen(true)}
        />
      </div>

      <Suspense fallback={null}>
        <NewCaseDialog
          isOpen={isNewCaseOpen}
          onOpenChange={setIsNewCaseOpen}
          onCaseCreated={handleCaseCreated}
        />
      </Suspense>
    </div>
  );
}
