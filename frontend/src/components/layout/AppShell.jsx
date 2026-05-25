import React, { useState, useEffect, useCallback, lazy, Suspense } from 'react';
import { AppSidebar } from './AppSidebar';
import { AppHeader } from './AppHeader';
import { LandingHero } from './LandingHero';

const NewCaseDialog = lazy(() => import('../ingest/NewCaseDialog').then(m => ({ default: m.NewCaseDialog })));
const IngestDialog  = lazy(() => import('../ingest/IngestDialog').then(m => ({ default: m.IngestDialog })));
import { Sheet, SheetContent } from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { AlertTriangle, Upload } from 'lucide-react';
import * as api from '@/api';

export function AppShell({ activeCaseId, activeMode, children }) {
  const [cases, setCases] = useState([]);
  const [isNewCaseOpen, setIsNewCaseOpen] = useState(false);
  const [isIngestOpen, setIsIngestOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const fetchCases = useCallback(async () => {
    try {
      const resp = await api.getCases();
      setCases(resp.data?.cases || []);
    } catch (err) {
      console.error('Failed to fetch cases:', err);
    }
  }, []);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const activeCaseData = cases.find(c => c.case_id === activeCaseId);
  const hasNoSources = !!(
    activeCaseId &&
    activeCaseId !== 'empty' &&
    activeCaseData &&
    activeCaseData.total_chunks === 0
  );

  const handleCaseChange = (caseId) => {
    setMobileMenuOpen(false);
    window.location.hash = `#/case/${caseId}/${activeMode}`;
  };

  const handleModeChange = (mode) => {
    if (activeCaseId) {
      window.location.hash = `#/case/${activeCaseId}/${mode}`;
    } else {
      window.location.hash = `#/case/empty/${mode}`;
    }
  };

  const handleCaseCreated = (newCaseName) => {
    fetchCases();
    setMobileMenuOpen(false);
    window.location.hash = `#/case/${newCaseName}/${activeMode || 'query'}`;
  };

  const handleDeleteCase = async (id) => {
    try {
      await api.deleteCase(id);
      setCases(cases.filter(c => (typeof c === 'string' ? c : (c.case_id || c.id || c.name)) !== id));
      if (activeCaseId === id) {
        window.location.hash = `#/case/empty/${activeMode}`;
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <div className="hidden lg:block h-screen print-hide">
        <AppSidebar
          cases={cases}
          activeCaseId={activeCaseId}
          onCaseSelect={handleCaseChange}
          onCreateNew={() => setIsNewCaseOpen(true)}
          onAddSource={() => setIsIngestOpen(true)}
          onDeleteCase={handleDeleteCase}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
        />
      </div>

      <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
        <SheetContent side="left" className="p-0 w-[240px] border-r-0 print-hide">
          <AppSidebar
            cases={cases}
            activeCaseId={activeCaseId}
            onCaseSelect={handleCaseChange}
            onCreateNew={() => setIsNewCaseOpen(true)}
            onAddSource={() => { setIsIngestOpen(true); setMobileMenuOpen(false); }}
            onDeleteCase={handleDeleteCase}
            collapsed={false}
          />
        </SheetContent>
      </Sheet>

      <div className="flex-1 flex flex-col min-w-0">
        <AppHeader
          cases={cases}
          activeCaseId={activeCaseId}
          onCaseChange={handleCaseChange}
          activeMode={activeMode}
          onModeChange={handleModeChange}
          onMobileMenuClick={() => setMobileMenuOpen(true)}
          className="print-hide"
        />
        <main className="flex-1 overflow-hidden flex flex-col">
          {hasNoSources && (
            <div className="flex items-center gap-3 px-4 py-2.5 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-300 shrink-0 print-hide">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <p className="text-sm flex-1">
                No sources found for this case. Ingest documents or videos to enable analysis.
              </p>
              <Button
                variant="outline"
                size="sm"
                className="border-amber-300 dark:border-amber-700 text-amber-800 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-900/50 shrink-0"
                onClick={() => setIsIngestOpen(true)}
              >
                <Upload className="w-3.5 h-3.5 mr-1.5" />
                Add source
              </Button>
            </div>
          )}
          <div className="flex-1 overflow-hidden">
            {(!activeCaseId || activeCaseId === 'empty')
              ? <LandingHero onCreateCase={() => setIsNewCaseOpen(true)} />
              : children
            }
          </div>
        </main>
      </div>

      <Suspense fallback={null}>
        <NewCaseDialog
          isOpen={isNewCaseOpen}
          onOpenChange={setIsNewCaseOpen}
          onCaseCreated={handleCaseCreated}
        />
        <IngestDialog
          isOpen={isIngestOpen}
          onOpenChange={setIsIngestOpen}
          activeCaseId={activeCaseId}
          onIngested={fetchCases}
        />
      </Suspense>
    </div>
  );
}
