import { useState } from 'react';
import { detectContradictions } from '@/api';
import { ModeFrame } from '@/components/shared/ModeFrame';
import { EmptyState } from '@/components/shared/EmptyState';
import { SkeletonCard } from '@/components/shared/SkeletonCard';
import { ContradictionCard } from '@/components/shared/ContradictionCard';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

// Maps new dict API format → ContradictionCard props
function toCardProps(c) {
  return {
    severity:          c.severity,
    speaker_a:         c.speaker_a,
    timestamp_a:       c.timestamp_a,
    source_name_a:     c.source_name_a,
    speaker_b:         c.speaker_b,
    timestamp_b:       c.timestamp_b,
    source_name_b:     c.source_name_b,
    claim:             c.statement_a,
    claim_citation:    { ref: c.citation_a, type: 'transcript' },
    evidence:          c.statement_b,
    evidence_citation: { ref: c.citation_b, type: 'transcript' },
    explanation:       c.explanation,
    is_cross_source:   c.is_cross_source,
  };
}

const SEVERITY_TABS = ['all', 'high', 'medium', 'low'];

export default function ContradictionsMode({ caseId }) {
  const [contradictions, setContradictions] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('all');

  const run = async () => {
    if (!caseId) { toast.error('No case selected'); return; }
    setLoading(true);
    try {
      const res = await detectContradictions(caseId);
      setContradictions(res.data.contradictions);
      setSummary(res.data.summary);
      setActiveTab('all');
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const countFor = (sev) =>
    sev === 'all'
      ? contradictions.length
      : contradictions.filter(c => c.severity === sev).length;

  const filtered =
    activeTab === 'all'
      ? contradictions
      : contradictions.filter(c => c.severity === activeTab);

  const crossCount = contradictions.filter(c => c.is_cross_source).length;

  return (
    <ModeFrame
      title="Contradictions"
      subtitle={caseId || 'No case selected'}
      actions={
        <Button onClick={run} disabled={loading || !caseId}>
          <RefreshCw className={cn('w-4 h-4 mr-2', loading && 'animate-spin')} />
          {loading ? 'Detecting…' : contradictions.length ? 'Re-run' : 'Detect contradictions'}
        </Button>
      }
    >
      {loading && (
        <div className="space-y-4">
          {[1, 2, 3].map(i => <SkeletonCard key={i} rows={3} />)}
        </div>
      )}

      {!loading && contradictions.length === 0 && (
        <EmptyState
          icon={AlertTriangle}
          title="Find contradictions"
          description="Runs a two-stage analysis: embedding similarity filter, then LLM verification across all case sources."
        />
      )}

      {!loading && summary && (
        <div className="space-y-6">
          {/* Stats row */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { label: 'Total', count: summary.total_contradictions ?? contradictions.length, cls: 'text-foreground' },
              { label: 'High', count: summary.high_severity, cls: 'text-destructive' },
              { label: 'Medium', count: summary.medium_severity, cls: 'text-warning' },
              { label: 'Low', count: summary.low_severity, cls: 'text-success' },
              { label: 'Cross-source', count: crossCount, cls: 'text-primary' },
              { label: 'Stmts checked', count: summary.statements_checked, cls: 'text-muted-foreground' },
            ].map(s => (
              <Card key={s.label}>
                <CardContent className="p-3">
                  <div className={cn('text-2xl font-bold tabular-nums', s.cls)}>{s.count}</div>
                  <div className="text-xs text-muted-foreground mt-0.5 leading-tight">{s.label}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Severity filter tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              {SEVERITY_TABS.map(tab => (
                <TabsTrigger key={tab} value={tab} className="capitalize">
                  {tab} ({countFor(tab)})
                </TabsTrigger>
              ))}
            </TabsList>

            <TabsContent value={activeTab} className="space-y-4 mt-4">
              {filtered.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No contradictions at this severity level.
                </p>
              )}
              {filtered.map((c, i) => (
                <ContradictionCard key={i} contradiction={toCardProps(c)} />
              ))}
            </TabsContent>
          </Tabs>
        </div>
      )}
    </ModeFrame>
  );
}
