import { useState } from 'react';
import { evidenceMap } from '@/api';
import { ModeFrame } from '@/components/shared/ModeFrame';
import { EmptyState } from '@/components/shared/EmptyState';
import { SkeletonCard } from '@/components/shared/SkeletonCard';
import { CitationBadge } from '@/components/shared/CitationBadge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Scale, Search, Edit2 } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const STRENGTH_STYLES = {
  strong: 'bg-success/10 text-success border-success/30',
  moderate: 'bg-warning/10 text-warning border-warning/30',
  weak: 'bg-destructive/10 text-destructive border-destructive/30',
};

const COLUMN_CONFIG = [
  { key: 'supporting', label: 'Supporting', borderColor: 'hsl(var(--success))' },
  { key: 'opposing', label: 'Opposing', borderColor: 'hsl(var(--destructive))' },
  { key: 'neutral', label: 'Neutral', borderColor: 'hsl(var(--muted-foreground))' },
];

function StrengthBadge({ strength }) {
  let norm;
  if (typeof strength === 'number') {
    if (strength >= 4) norm = 'strong';
    else if (strength === 3) norm = 'moderate';
    else norm = 'weak';
  } else {
    norm = (strength || '').toString().toLowerCase();
  }
  const label = typeof strength === 'number'
    ? `${norm.charAt(0).toUpperCase() + norm.slice(1)} (${strength})`
    : strength;
  return (
    <Badge variant="outline" className={STRENGTH_STYLES[norm] || 'bg-muted text-muted-foreground'}>
      {label}
    </Badge>
  );
}

// API row format: [strength, citation_ref, ?, reason, text]
function EvidenceCard({ row, borderColor }) {
  return (
    <Card className="border-l-[4px]" style={{ borderLeftColor: borderColor }}>
      <CardContent className="p-4 space-y-2">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <CitationBadge citation={{ citation_ref: row[1], type: row[2] || 'video' }} />
          <StrengthBadge strength={row[0]} />
        </div>
        {row[3] && <p className="text-sm text-muted-foreground">{row[3]}</p>}
        {row[4] && <p className="text-sm">{row[4]}</p>}
      </CardContent>
    </Card>
  );
}

export default function EvidenceMode({ caseId }) {
  const [position, setPosition] = useState('');
  const [committedPosition, setCommittedPosition] = useState('');
  const [data, setData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = async (pos) => {
    const p = pos ?? position;
    if (!caseId || !p.trim()) {
      toast.error(!caseId ? 'No case selected' : 'Enter a legal position');
      return;
    }
    setLoading(true);
    try {
      const res = await evidenceMap(caseId, p);
      setData(res.data);
      setSummary(res.data.summary);
      setCommittedPosition(p);
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setData(null);
    setSummary(null);
    setCommittedPosition('');
  };

  const positionBar = data && (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground truncate max-w-[200px] hidden sm:inline">
        {committedPosition}
      </span>
      <Button variant="outline" size="sm" onClick={reset}>
        <Edit2 className="w-3 h-3 mr-2" />
        Edit position
      </Button>
    </div>
  );

  return (
    <ModeFrame
      title="Evidence Map"
      subtitle={caseId || 'No case selected'}
      actions={positionBar}
    >
      {!data && !loading && (
        <div className="max-w-2xl mx-auto pt-8 space-y-6">
          <EmptyState
            icon={Scale}
            title="Map your evidence"
            description="Enter your legal position to classify all case evidence as supporting, opposing, or neutral."
          />
          <div className="flex gap-2">
            <Input
              value={position}
              onChange={e => setPosition(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && run()}
              placeholder="e.g. Defending driver against negligence charges"
              className="flex-1"
            />
            <Button onClick={() => run()} disabled={!position.trim()}>
              <Search className="w-4 h-4 mr-2" />
              Map evidence
            </Button>
          </div>
        </div>
      )}

      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <SkeletonCard rows={4} />
          <SkeletonCard rows={4} />
          <SkeletonCard rows={4} />
        </div>
      )}

      {data && summary && !loading && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Supporting', count: summary.supporting_count, cls: 'text-success' },
              { label: 'Opposing', count: summary.opposing_count, cls: 'text-destructive' },
              { label: 'Neutral', count: summary.neutral_count, cls: 'text-muted-foreground' },
              { label: 'Total', count: summary.total_classified, cls: 'text-foreground' },
            ].map(s => (
              <Card key={s.label}>
                <CardContent className="p-4">
                  <div className={cn('text-3xl font-bold tabular-nums', s.cls)}>{s.count}</div>
                  <div className="text-sm text-muted-foreground mt-1">{s.label}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Desktop: 3 columns */}
          <div className="hidden lg:grid grid-cols-3 gap-4">
            {COLUMN_CONFIG.map(col => (
              <div key={col.key} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">{col.label}</h3>
                  <Badge variant="secondary">{data[col.key]?.length ?? 0}</Badge>
                </div>
                {(data[col.key]?.length ?? 0) === 0 && (
                  <p className="text-sm text-muted-foreground">None found.</p>
                )}
                {data[col.key]?.map((row, i) => (
                  <EvidenceCard key={i} row={row} borderColor={col.borderColor} />
                ))}
              </div>
            ))}
          </div>

          {/* Mobile: tabs */}
          <div className="lg:hidden">
            <Tabs defaultValue="supporting">
              <TabsList className="w-full">
                {COLUMN_CONFIG.map(col => (
                  <TabsTrigger key={col.key} value={col.key} className="flex-1 text-xs sm:text-sm">
                    {col.label} ({data[col.key]?.length ?? 0})
                  </TabsTrigger>
                ))}
              </TabsList>
              {COLUMN_CONFIG.map(col => (
                <TabsContent key={col.key} value={col.key} className="space-y-3 mt-4">
                  {(data[col.key]?.length ?? 0) === 0 && (
                    <p className="text-sm text-muted-foreground">None found.</p>
                  )}
                  {data[col.key]?.map((row, i) => (
                    <EvidenceCard key={i} row={row} borderColor={col.borderColor} />
                  ))}
                </TabsContent>
              ))}
            </Tabs>
          </div>
        </div>
      )}
    </ModeFrame>
  );
}
