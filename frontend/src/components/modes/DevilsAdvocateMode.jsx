import { useState } from 'react';
import { daNewSession, daArgue } from '@/api';
import { ModeFrame } from '@/components/shared/ModeFrame';
import { EmptyState } from '@/components/shared/EmptyState';
import { SkeletonCard } from '@/components/shared/SkeletonCard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Swords, Plus, FileText, ShieldAlert, Zap, TrendingUp, Target } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

// Normalise weaknesses — handles both List[Dict] (new) and List[str] (old)
function normaliseWeaknesses(raw) {
  if (!raw || !Array.isArray(raw)) return [];
  return raw.map(w => {
    if (typeof w === 'string') return { point: w, exhibit: '', severity: 'medium' };
    return {
      point:    String(w.point || ''),
      exhibit:  String(w.exhibit || ''),
      severity: String(w.severity || 'medium').toLowerCase(),
    };
  }).filter(w => w.point);
}

const SEV_BADGE = {
  high:   'border-destructive/50 text-destructive',
  medium: 'border-warning/50 text-warning',
  low:    'border-muted text-muted-foreground',
};

function RoundCard({ round, number, isActive }) {
  const critiqueText    = round.critique || '';
  const weaknesses      = normaliseWeaknesses(round.weaknesses);
  const counterArgument = round.counter_argument || round.opposition || '';
  const defensePct      = round.defense_probability != null
    ? Math.round(round.defense_probability * 100)
    : null;
  const strategy        = round.recommended_defense_strategy || '';
  const lawyerArg       = round.lawyer_argument || round.argument || '';

  return (
    <Card className={cn('overflow-hidden transition-shadow', isActive && 'ring-2 ring-primary shadow-lg')}>
      {/* Round header */}
      <CardHeader className="py-3 px-4 bg-muted/40 border-b border-border flex flex-row items-center justify-between space-y-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Round {number}</span>
          {isActive && <Badge variant="secondary" className="text-xs">Current</Badge>}
        </div>
        {round.evidence_count > 0 && (
          <Badge variant="outline" className="text-xs text-muted-foreground">
            <FileText className="w-3 h-3 mr-1" />
            {round.evidence_count} sources cited
          </Badge>
        )}
      </CardHeader>

      <CardContent className="p-4 space-y-4">
        {/* Lawyer's argument */}
        {lawyerArg && (
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wider text-primary">Your argument</p>
            <p className="text-sm bg-primary/5 border border-primary/10 rounded-md px-3 py-2 leading-relaxed">
              {lawyerArg}
            </p>
          </div>
        )}

        {/* Critique — red bordered card */}
        {critiqueText && (
          <div className="border border-destructive/30 rounded-lg p-3 bg-destructive/5">
            <div className="flex items-center gap-1.5 mb-2">
              <ShieldAlert className="w-3.5 h-3.5 text-destructive" />
              <p className="text-xs font-semibold uppercase tracking-wider text-destructive">Critique</p>
            </div>
            <p className="text-sm leading-relaxed">{critiqueText}</p>
          </div>
        )}

        {/* Weaknesses — list with severity badge + exhibit tag */}
        {weaknesses.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-warning" />
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Weaknesses found
              </p>
            </div>
            <ul className="space-y-2">
              {weaknesses.map((w, i) => (
                <li key={i} className="flex gap-2 items-start text-sm">
                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 bg-warning/70" />
                  <span className="flex-1 text-muted-foreground leading-snug">{w.point}</span>
                  <div className="flex gap-1 shrink-0 flex-wrap justify-end">
                    {w.exhibit && (
                      <Badge variant="outline" className="text-xs font-mono">{w.exhibit}</Badge>
                    )}
                    {w.severity && (
                      <Badge variant="outline" className={cn('text-xs', SEV_BADGE[w.severity] || SEV_BADGE.low)}>
                        {w.severity}
                      </Badge>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        <Separator />

        {/* Counter argument — blue bordered card */}
        {counterArgument && (
          <div className="border border-primary/30 rounded-lg p-3 bg-primary/5">
            <div className="flex items-center gap-1.5 mb-2">
              <Swords className="w-3.5 h-3.5 text-primary" />
              <p className="text-xs font-semibold uppercase tracking-wider text-primary">Counter Argument</p>
            </div>
            <p className="text-sm leading-relaxed">{counterArgument}</p>
          </div>
        )}

        {/* Defense probability gauge */}
        {defensePct != null && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold uppercase tracking-wider text-muted-foreground">Defense probability</span>
              <span className="font-bold tabular-nums">{defensePct}%</span>
            </div>
            <Progress value={defensePct} className="h-2" />
          </div>
        )}

        {/* Recommended defense strategy — highlighted action card */}
        {strategy && (
          <div className="bg-success/10 border border-success/30 rounded-lg p-3">
            <div className="flex items-center gap-1.5 mb-2">
              <Target className="w-3.5 h-3.5 text-success" />
              <p className="text-xs font-semibold uppercase tracking-wider text-success">Recommended Strategy</p>
            </div>
            <p className="text-sm leading-relaxed text-success/90">{strategy}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function DevilsAdvocateMode({ caseId }) {
  const [topic, setTopic]         = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [argument, setArgument]   = useState('');
  const [rounds, setRounds]       = useState([]);
  const [loading, setLoading]     = useState(false);

  const startSession = async () => {
    if (!caseId || !topic.trim()) {
      toast.error(!caseId ? 'No case selected' : 'Enter a topic');
      return;
    }
    setLoading(true);
    try {
      const res = await daNewSession(caseId, topic);
      setSessionId(res.data.session_id);
      setRounds([]);
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const submitArgument = async () => {
    if (!argument.trim() || !sessionId) return;
    setLoading(true);
    try {
      const res = await daArgue(sessionId, caseId, argument);
      setRounds(prev => [...prev, res.data]);
      setArgument('');
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const newTopic = () => {
    setSessionId(null);
    setTopic('');
    setRounds([]);
    setArgument('');
  };

  const actions = sessionId ? (
    <div className="flex items-center gap-2">
      <Badge variant="outline" className="font-normal">Round {rounds.length + 1}</Badge>
      <Button variant="ghost" size="sm" onClick={newTopic}>
        <Plus className="w-4 h-4 mr-1.5" />New topic
      </Button>
    </div>
  ) : null;

  return (
    <ModeFrame
      title="Devil's Advocate"
      subtitle={sessionId ? `Topic: ${topic}` : (caseId || 'No case selected')}
      actions={actions}
    >
      {!sessionId ? (
        <div className="max-w-xl mx-auto pt-8 space-y-6">
          <EmptyState
            icon={Swords}
            title="Challenge your arguments"
            description="State a legal argument. The system attacks it using actual case evidence — finds weaknesses, surfaces what opposing counsel will argue, and recommends a defense strategy."
          />
          <div className="space-y-3">
            <Input
              value={topic}
              onChange={e => setTopic(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && startSession()}
              placeholder="e.g. Gearbox fault defence strategy"
              autoFocus
            />
            <Button className="w-full" onClick={startSession} disabled={loading || !topic.trim()}>
              {loading ? 'Starting…' : 'Start session'}
            </Button>
          </div>
        </div>
      ) : (
        <div className="max-w-3xl mx-auto space-y-6">
          {rounds.map((round, i) => (
            <RoundCard key={i} round={round} number={i + 1} isActive={i === rounds.length - 1} />
          ))}

          {loading && <SkeletonCard rows={6} />}

          {rounds.length === 0 && !loading && (
            <p className="text-sm text-muted-foreground text-center py-6">
              Submit your first argument below — the system will attack it using evidence from the case.
            </p>
          )}

          <Card>
            <CardContent className="p-4 space-y-3">
              <p className="text-sm font-medium text-muted-foreground">
                Your argument — Round {rounds.length + 1}
              </p>
              <Textarea
                value={argument}
                onChange={e => setArgument(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) submitArgument(); }}
                placeholder="State your legal argument…"
                rows={4}
              />
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">⌘↵ to submit</p>
                <Button onClick={submitArgument} disabled={loading || !argument.trim()}>
                  {loading ? 'Analysing…' : 'Submit argument'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </ModeFrame>
  );
}
