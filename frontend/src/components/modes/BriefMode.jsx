import { useState } from 'react';
import { generateBrief } from '@/api';
import { ModeFrame } from '@/components/shared/ModeFrame';
import { EmptyState } from '@/components/shared/EmptyState';
import { SkeletonCard } from '@/components/shared/SkeletonCard';
import { ContradictionCard } from '@/components/shared/ContradictionCard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  FileText, Printer, Edit2,
  ShieldAlert, TrendingUp, Users, AlertTriangle,
  Swords, HelpCircle, CheckCircle2, ChevronDown, ChevronUp,
  BookOpen, Target,
} from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

// ── Constants ─────────────────────────────────────────────────────

const STRENGTH_STYLE = {
  STRONG:   { text: 'text-success',     stroke: 'hsl(var(--success))' },
  MODERATE: { text: 'text-warning',     stroke: 'hsl(var(--warning))' },
  WEAK:     { text: 'text-destructive', stroke: 'hsl(var(--destructive))' },
};
const RELIABILITY_STYLE = {
  FRIENDLY: 'bg-success/10 text-success border-success/30',
  NEUTRAL:  'bg-warning/10 text-warning border-warning/30',
  HOSTILE:  'bg-destructive/10 text-destructive border-destructive/30',
};
const PRIORITY_STYLE = {
  high:   'text-destructive border-destructive/50',
  medium: 'text-warning border-warning/50',
  low:    'text-success border-success/50',
};

// ── LLM output coercers ───────────────────────────────────────────

function renderSummary(summary) {
  if (!summary) return [];
  let obj = null;
  if (typeof summary === 'string') {
    try { obj = JSON.parse(summary); } catch {}
  } else if (typeof summary === 'object') {
    obj = summary;
  }
  if (obj && typeof obj === 'object') {
    const paras = ['paragraph_1', 'paragraph_2', 'paragraph_3']
      .map(k => (typeof obj[k] === 'object' ? obj[k]?.text : obj[k]))
      .filter(Boolean);
    if (paras.length) return paras;
    const vals = Object.values(obj).filter(v => typeof v === 'string' && v.trim());
    if (vals.length) return vals;
  }
  return String(summary).split(/\n{2,}/).map(s => s.trim()).filter(Boolean);
}

function toStringList(val) {
  if (!val) return [];
  if (Array.isArray(val)) {
    return val.map(v => (typeof v === 'string' ? v : v?.action || v?.question || JSON.stringify(v))).filter(Boolean);
  }
  if (typeof val === 'string') {
    try {
      const p = JSON.parse(val);
      if (Array.isArray(p)) return p.filter(Boolean);
    } catch {}
    return val.split('\n').map(s => s.replace(/^[-•*\d.]\s*/, '')).filter(Boolean);
  }
  return [];
}

// ── Helpers ───────────────────────────────────────────────────────

function DonutRing({ score, stroke }) {
  const r = 44, circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  return (
    <svg viewBox="0 0 100 100" className="w-24 h-24 shrink-0">
      <circle cx="50" cy="50" r={r} fill="none" stroke="hsl(var(--muted))" strokeWidth="10" />
      <circle cx="50" cy="50" r={r} fill="none" stroke={stroke} strokeWidth="10"
        strokeLinecap="round" strokeDasharray={`${dash} ${circ}`}
        transform="rotate(-90 50 50)" />
      <text x="50" y="56" textAnchor="middle" fill={stroke}
        style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'ui-sans-serif, system-ui, sans-serif' }}>
        {score}
      </text>
    </svg>
  );
}

function NumberedList({ items, accent }) {
  if (!items?.length) return null;
  return (
    <ol className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex gap-2 text-sm">
          <span className={cn('font-semibold shrink-0 tabular-nums', accent ? 'text-success' : 'text-primary')}>
            {i + 1}.
          </span>
          <span className="text-muted-foreground leading-snug">{item}</span>
        </li>
      ))}
    </ol>
  );
}

function EvidenceRow({ item, variant }) {
  const borderColor = variant === 'strong' ? 'hsl(var(--success))' : 'hsl(var(--destructive))';
  return (
    <div className="border-l-[3px] pl-3 py-1 space-y-1" style={{ borderLeftColor: borderColor }}>
      <div className="flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="text-xs font-mono">{item.citation || item.exhibit}</Badge>
        {item.strength && <span className="text-xs text-muted-foreground">{item.strength}</span>}
      </div>
      {item.reason && <p className="text-xs text-muted-foreground">{item.reason}</p>}
      {item.significance && <p className="text-xs text-muted-foreground">{item.significance}</p>}
      {item.text && <p className="text-sm">{item.text}</p>}
      {item.summary && !item.text && <p className="text-sm text-muted-foreground">{item.summary}</p>}
    </div>
  );
}

// Key evidence card with exhibit badge + significance
function KeyEvidenceCard({ item }) {
  return (
    <div className="border rounded-lg p-3 space-y-1.5 bg-muted/20">
      <div className="flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="text-xs font-mono">{item.exhibit}</Badge>
        {item.significance && (
          <Badge variant="secondary" className="text-xs">{item.significance.slice(0, 40)}</Badge>
        )}
      </div>
      {item.summary && <p className="text-sm text-muted-foreground leading-snug">{item.summary}</p>}
    </div>
  );
}

// Recommended action row with priority color coding
function ActionRow({ action }) {
  if (typeof action === 'string') {
    return (
      <li className="flex gap-2 text-sm">
        <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0 text-primary" />
        <span className="text-muted-foreground leading-snug">{action}</span>
      </li>
    );
  }
  const pri = (action.priority || 'medium').toLowerCase();
  return (
    <li className="flex gap-2 text-sm items-start">
      <Badge variant="outline" className={cn('text-xs shrink-0 mt-0.5', PRIORITY_STYLE[pri] || PRIORITY_STYLE.medium)}>
        {pri}
      </Badge>
      <div className="flex-1 space-y-0.5">
        <p className="leading-snug">{action.action}</p>
        {action.rationale && (
          <p className="text-xs text-muted-foreground">{action.rationale}</p>
        )}
      </div>
    </li>
  );
}

// Cross-examination question with exhibit tag + strategic goal
function CrossExamRow({ item, number }) {
  if (typeof item === 'string') {
    return (
      <li className="flex gap-2 text-sm">
        <span className="font-semibold shrink-0 tabular-nums text-primary">{number}.</span>
        <span className="text-muted-foreground leading-snug">{item}</span>
      </li>
    );
  }
  return (
    <li className="space-y-1 text-sm">
      <div className="flex gap-2 items-start">
        <span className="font-semibold shrink-0 tabular-nums text-primary">{number}.</span>
        <span className="leading-snug">{item.question}</span>
        {item.target_exhibit && (
          <Badge variant="outline" className="text-xs font-mono shrink-0 ml-auto">{item.target_exhibit}</Badge>
        )}
      </div>
      {item.strategic_goal && (
        <p className="text-xs text-muted-foreground pl-5">{item.strategic_goal}</p>
      )}
    </li>
  );
}

// Collapsible weakness card
function WeaknessCard({ item }) {
  const [open, setOpen] = useState(false);
  const weakness   = typeof item === 'string' ? item : item.weakness;
  const mitigation = typeof item === 'object' ? item.mitigation : '';
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center justify-between p-3 text-left hover:bg-muted/40 transition-colors"
        onClick={() => setOpen(v => !v)}
      >
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-destructive shrink-0" />
          <span className="text-sm leading-snug">{weakness}</span>
        </div>
        {mitigation && (open ? <ChevronUp className="w-4 h-4 shrink-0" /> : <ChevronDown className="w-4 h-4 shrink-0" />)}
      </button>
      {open && mitigation && (
        <div className="px-3 pb-3 pt-0 bg-success/5 border-t border-border">
          <p className="text-xs text-muted-foreground">
            <span className="font-semibold text-success">Mitigation: </span>{mitigation}
          </p>
        </div>
      )}
    </div>
  );
}

// Contradiction with conflict score bar
function ContradictionSummaryRow({ item }) {
  const pct = Math.round((item.conflict_score || 0) * 100);
  return (
    <div className="space-y-1.5 border-l-[3px] border-destructive/40 pl-3 py-1">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-mono text-muted-foreground">{item.exhibits}</span>
        <Badge variant="outline" className="text-xs border-destructive/40 text-destructive">
          score {item.conflict_score?.toFixed(2)}
        </Badge>
      </div>
      <Progress value={pct} className="h-1.5" />
      {item.impact && <p className="text-xs text-muted-foreground">{item.impact}</p>}
    </div>
  );
}

function WitnessCard({ witness }) {
  const [expanded, setExpanded] = useState(false);
  const initials      = (witness.speaker_id || '').substring(0, 2).toUpperCase();
  const reliabilityCls = RELIABILITY_STYLE[witness.reliability_rating] || 'bg-muted text-muted-foreground border-muted';
  const credPct        = Math.round((witness.credibility_score / 10) * 100);
  const hasExtra       = (witness.key_statements?.length > 0) || (witness.suggested_questions?.length > 0);

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm font-semibold shrink-0">
            {initials}
          </div>
          <div className="min-w-0">
            <p className="font-medium text-sm truncate">{witness.speaker_id}</p>
            <p className="text-xs text-muted-foreground truncate">{witness.inferred_role}</p>
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 flex-wrap">
          <Badge variant="outline" className={reliabilityCls}>{witness.reliability_rating}</Badge>
          <span className="text-xs text-muted-foreground tabular-nums">
            {witness.contradiction_count} contradiction{witness.contradiction_count !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Credibility</span>
            <span className="tabular-nums font-medium">{witness.credibility_score}/10</span>
          </div>
          <Progress value={credPct} className="h-1.5" />
        </div>

        {witness.recommended_approach && (
          <p className="text-xs text-muted-foreground border-t border-border pt-3 leading-relaxed">
            {witness.recommended_approach}
          </p>
        )}

        {hasExtra && (
          <>
            <button
              onClick={() => setExpanded(v => !v)}
              className="flex items-center gap-1 text-xs text-primary hover:underline"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {expanded ? 'Show less' : 'Key statements & questions'}
            </button>
            {expanded && (
              <div className="space-y-3 border-t border-border pt-3">
                {witness.key_statements?.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Key statements</p>
                    <ul className="space-y-1">
                      {witness.key_statements.map((s, i) => (
                        <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                          <span className="mt-1.5 w-1 h-1 rounded-full bg-muted-foreground/50 shrink-0" />
                          {s}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {witness.suggested_questions?.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Suggested questions</p>
                    <ol className="space-y-1">
                      {witness.suggested_questions.map((q, i) => (
                        <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                          <span className="shrink-0 font-medium text-primary">{i + 1}.</span>
                          {q}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function SectionCard({ icon: Icon, title, iconClass, children }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Icon className={cn('w-4 h-4', iconClass)} />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

// ── Main component ────────────────────────────────────────────────

export default function BriefMode({ caseId }) {
  const [position, setPosition]   = useState('');
  const [committed, setCommitted] = useState('');
  const [brief, setBrief]         = useState(null);
  const [loading, setLoading]     = useState(false);

  const run = async () => {
    if (!caseId || !position.trim()) {
      toast.error(!caseId ? 'No case selected' : 'Enter a legal position');
      return;
    }
    setLoading(true);
    try {
      const res = await generateBrief(caseId, position);
      setBrief(res.data);
      setCommitted(position);
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setBrief(null); setCommitted(''); };

  const strength     = (brief?.case_strength || '').toUpperCase();
  // Use numeric score if available, otherwise fall back to label-based default
  const score        = brief?.case_strength_score ?? ({ STRONG: 82, MODERATE: 54, WEAK: 22 }[strength] ?? 50);
  const style        = STRENGTH_STYLE[strength] || { text: 'text-muted-foreground', stroke: 'hsl(var(--muted-foreground))' };

  const assessmentText = brief?.assessment || brief?.overall_assessment || '';

  const briefContradictions = (brief?.contradictions || []).map(c => ({
    severity:          (c.severity || '').toLowerCase(),
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
  }));

  const actions = brief ? (
    <div className="flex items-center gap-2">
      <Button variant="outline" size="sm" onClick={reset}>
        <Edit2 className="w-3 h-3 mr-2" />Edit position
      </Button>
      <Button variant="outline" size="sm" onClick={() => window.print()}>
        <Printer className="w-4 h-4 mr-2" />Print / Save PDF
      </Button>
    </div>
  ) : null;

  return (
    <ModeFrame title="Trial Brief" subtitle={caseId || 'No case selected'} actions={actions} className="print-root">
      {/* Empty / input state */}
      {!brief && !loading && (
        <div className="max-w-2xl mx-auto pt-8 space-y-6">
          <EmptyState
            icon={FileText}
            title="Generate a trial brief"
            description="Compiles witness profiles, contradictions, evidence map, and strategy from all case sources — grounded in real indexed evidence."
          />
          <div className="flex gap-2">
            <Input
              value={position}
              onChange={e => setPosition(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && run()}
              placeholder="Your legal position…"
              className="flex-1"
              autoFocus
            />
            <Button onClick={run} disabled={!position.trim()}>Generate brief</Button>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="space-y-4">
          <SkeletonCard rows={3} />
          <SkeletonCard rows={5} />
          <SkeletonCard rows={4} />
        </div>
      )}

      {/* Brief content */}
      {brief && !loading && (
        <div className="space-y-6 brief-content">

          {/* Strength hero */}
          <Card>
            <CardContent className="p-6 flex items-center gap-6 flex-wrap">
              <DonutRing score={score} stroke={style.stroke} />
              <div className="min-w-0 flex-1">
                <div className={cn('text-2xl font-bold', style.text)}>{strength} CASE</div>
                {assessmentText && (
                  <p className="text-sm text-muted-foreground mt-1 leading-relaxed max-w-prose">
                    {assessmentText}
                  </p>
                )}
                <p className="text-xs text-muted-foreground mt-2">
                  Brief {brief.brief_id} · {committed}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Case summary — paragraph_1/2/3 */}
          {brief.case_summary && (
            <SectionCard icon={FileText} title="Case Summary" iconClass="text-muted-foreground">
              <div className="space-y-3">
                {renderSummary(brief.case_summary).map((para, i) => (
                  <p key={i} className="text-sm text-muted-foreground leading-relaxed">{para}</p>
                ))}
              </div>
            </SectionCard>
          )}

          {/* Key evidence — exhibit cards */}
          {brief.key_evidence?.length > 0 && (
            <SectionCard icon={BookOpen} title="Key Evidence" iconClass="text-primary">
              <div className="space-y-2">
                {brief.key_evidence.map((item, i) => (
                  <KeyEvidenceCard key={i} item={item} />
                ))}
              </div>
            </SectionCard>
          )}

          {/* Key Risks */}
          {toStringList(brief.key_risks).length > 0 && (
            <SectionCard icon={AlertTriangle} title="Key Risks" iconClass="text-destructive">
              <NumberedList items={toStringList(brief.key_risks)} />
            </SectionCard>
          )}

          {/* Recommended actions with priority color coding */}
          {brief.recommended_actions?.length > 0 && (
            <SectionCard icon={CheckCircle2} title="Recommended Actions" iconClass="text-primary">
              <ul className="space-y-3">
                {brief.recommended_actions.map((action, i) => (
                  <ActionRow key={i} action={action} />
                ))}
              </ul>
            </SectionCard>
          )}

          {/* Cross-examination questions with target exhibit + strategic goal */}
          {(brief.cross_examination_detail?.length > 0 || toStringList(brief.questions_to_ask).length > 0) && (
            <SectionCard icon={HelpCircle} title="Cross-Examination Questions" iconClass="text-primary">
              <ol className="space-y-3">
                {(brief.cross_examination_detail?.length
                  ? brief.cross_examination_detail
                  : toStringList(brief.questions_to_ask)
                ).map((q, i) => (
                  <CrossExamRow key={i} item={q} number={i + 1} />
                ))}
              </ol>
            </SectionCard>
          )}

          {/* Opposition strategy */}
          {toStringList(brief.opposition_strategy).length > 0 && (
            <SectionCard icon={Swords} title="What Opposition Will Argue" iconClass="text-destructive">
              <NumberedList items={toStringList(brief.opposition_strategy)} />
            </SectionCard>
          )}

          {/* Contradictions summary with score bars */}
          {brief.contradictions_summary_detail?.length > 0 && (
            <SectionCard icon={AlertTriangle} title="Contradiction Scores" iconClass="text-destructive">
              <div className="space-y-3">
                {brief.contradictions_summary_detail.map((item, i) => (
                  <ContradictionSummaryRow key={i} item={item} />
                ))}
              </div>
            </SectionCard>
          )}

          {/* 2-col: Strongest evidence + Vulnerabilities */}
          {(brief.strongest_arguments?.length > 0 || brief.vulnerabilities?.length > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {brief.strongest_arguments?.length > 0 && (
                <SectionCard icon={TrendingUp} title="Strongest Evidence" iconClass="text-success">
                  <div className="space-y-3">
                    {brief.strongest_arguments.map((e, i) => (
                      <EvidenceRow key={i} item={e} variant="strong" />
                    ))}
                  </div>
                </SectionCard>
              )}
              {brief.vulnerabilities?.length > 0 && (
                <SectionCard icon={ShieldAlert} title="Vulnerabilities" iconClass="text-destructive">
                  <div className="space-y-3">
                    {brief.vulnerabilities.map((e, i) => (
                      <EvidenceRow key={i} item={e} variant="weak" />
                    ))}
                  </div>
                </SectionCard>
              )}
            </div>
          )}

          {/* Weaknesses as collapsible cards */}
          {brief.brief_weaknesses?.length > 0 && (
            <SectionCard icon={Target} title="Case Weaknesses & Mitigations" iconClass="text-warning">
              <div className="space-y-2">
                {brief.brief_weaknesses.map((w, i) => (
                  <WeaknessCard key={i} item={w} />
                ))}
              </div>
            </SectionCard>
          )}

          {/* Witness profiles */}
          {brief.witness_profiles?.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-muted-foreground" />
                <h2 className="text-base font-semibold">Witness Profiles</h2>
                <Badge variant="secondary">{brief.witness_profiles.length}</Badge>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {brief.witness_profiles.map((w, i) => (
                  <WitnessCard key={i} witness={w} />
                ))}
              </div>
            </div>
          )}

          {/* Contradictions */}
          {briefContradictions.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-destructive" />
                <h2 className="text-base font-semibold">Key Contradictions</h2>
                <Badge variant="secondary">{briefContradictions.length}</Badge>
              </div>
              <div className="space-y-3">
                {briefContradictions.map((c, i) => (
                  <ContradictionCard key={i} contradiction={c} />
                ))}
              </div>
            </div>
          )}

        </div>
      )}
    </ModeFrame>
  );
}
