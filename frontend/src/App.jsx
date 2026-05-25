import React, { useState, useEffect, lazy, Suspense } from 'react';
import { AppShell } from '@/components/layout/AppShell';
import { Toaster } from '@/components/ui/sonner';

const QueryMode         = lazy(() => import('@/components/modes/QueryMode'));
const EvidenceMode      = lazy(() => import('@/components/modes/EvidenceMode'));
const ContradictionsMode = lazy(() => import('@/components/modes/ContradictionsMode'));
const DevilsAdvocateMode = lazy(() => import('@/components/modes/DevilsAdvocateMode'));
const BriefMode         = lazy(() => import('@/components/modes/BriefMode'));

function ModeFallback() {
  return (
    <div className="flex-1 flex items-center justify-center h-full">
      <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

export default function App() {
  const [activeCaseId, setActiveCaseId] = useState('');
  const [activeMode, setActiveMode] = useState('query');

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash || '#/case/empty/query';
      const parts = hash.replace('#/', '').split('/');
      // Expected format: #/case/:caseId/:mode
      if (parts[0] === 'case') {
        const cId = parts[1] === 'empty' ? '' : decodeURIComponent(parts[1]);
        const mId = parts[2] || 'query';
        setActiveCaseId(cId);
        setActiveMode(mId);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    // Initialize
    if (!window.location.hash) {
      window.location.hash = '#/case/empty/query';
    } else {
      handleHashChange();
    }

    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const renderMode = () => {
    switch (activeMode) {
      case 'query':           return <QueryMode caseId={activeCaseId} />;
      case 'evidence':        return <EvidenceMode caseId={activeCaseId} />;
      case 'contradictions':  return <ContradictionsMode caseId={activeCaseId} />;
      case 'devils-advocate': return <DevilsAdvocateMode caseId={activeCaseId} />;
      case 'brief':           return <BriefMode caseId={activeCaseId} />;
      default:                return <QueryMode caseId={activeCaseId} />;
    }
  };

  return (
    <>
      <AppShell activeCaseId={activeCaseId} activeMode={activeMode}>
        <Suspense fallback={<ModeFallback />}>
          {renderMode()}
        </Suspense>
      </AppShell>
      <Toaster position="top-right" />
    </>
  );
}
