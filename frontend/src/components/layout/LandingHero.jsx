import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  MessageSquare, Scale, AlertTriangle,
  Swords, FileText, Upload, ChevronRight,
  Cpu, Database, Zap, Video, FileSearch
} from 'lucide-react';
import { cn } from '@/lib/utils';

const FEATURES = [
  {
    icon: MessageSquare,
    title: 'Query & Chat',
    description: 'Ask natural-language questions about the case. Multi-turn memory, persistent sessions, citations with exact timestamps and page numbers.',
    color: 'text-blue-500',
    bg: 'bg-blue-500/10',
  },
  {
    icon: Scale,
    title: 'Evidence Map',
    description: 'Classifies every piece of evidence as Supporting, Opposing, or Neutral relative to your legal position — instantly.',
    color: 'text-green-500',
    bg: 'bg-green-500/10',
  },
  {
    icon: AlertTriangle,
    title: 'Contradiction Detection',
    description: 'Cross-references all statements across documents and videos. Flags inconsistencies by severity with speaker attribution.',
    color: 'text-orange-500',
    bg: 'bg-orange-500/10',
  },
  {
    icon: Swords,
    title: "Devil's Advocate",
    description: 'Multi-round AI opponent that attacks your legal arguments using the actual case evidence — stress-tests before trial.',
    color: 'text-red-500',
    bg: 'bg-red-500/10',
  },
  {
    icon: FileText,
    title: 'Trial Brief',
    description: 'Generates a full brief: case strength score, witness credibility profiles, key risks, recommended actions, opposition strategy — as a PDF.',
    color: 'text-purple-500',
    bg: 'bg-purple-500/10',
  },
];

const STEPS = [
  {
    step: '01',
    icon: Upload,
    title: 'Upload your case files',
    description: 'PDFs (FIR, court orders, witness statements) or video depositions — even YouTube links.',
  },
  {
    step: '02',
    icon: Cpu,
    title: 'AI indexes everything',
    description: 'Multi-modal embeddings, WhisperX transcription, OCR — all chunked and stored in LanceDB.',
  },
  {
    step: '03',
    icon: Zap,
    title: 'Instant legal intelligence',
    description: 'Query, map evidence, detect contradictions, stress-test arguments, generate a trial brief.',
  },
];

const TECH = [
  { label: 'FastAPI', icon: '⚡' },
  { label: 'LanceDB', icon: '🗄️' },
  { label: 'CLIP + MiniLM', icon: '🧠' },
  { label: 'CrossEncoder Rerank', icon: '🎯' },
  { label: 'WhisperX', icon: '🎙️' },
  { label: 'Llama 3.1', icon: '🤖' },
];

export function LandingHero({ onCreateCase }) {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">

        {/* ── Hero ── */}
        <div className="text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium">
            <Scale className="w-3.5 h-3.5" />
            AI-Powered Legal Intelligence
          </div>

          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight leading-tight">
            From case files to{' '}
            <span className="text-primary">trial-ready strategy</span>
            <br className="hidden sm:block" /> in minutes.
          </h1>

          <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Upload your PDFs and video depositions. DIA-Legal indexes everything with
            multi-modal RAG, then gives you instant answers, evidence maps, contradiction
            detection, and a full trial brief — grounded in your actual files.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
            <Button size="lg" onClick={onCreateCase} className="gap-2 text-base px-6">
              <Upload className="w-4 h-4" />
              Create your first case
              <ChevronRight className="w-4 h-4" />
            </Button>
            <Button size="lg" variant="outline" className="gap-2 text-base px-6" asChild>
              <a href="https://github.com/asheesh07/DIA-Legal" target="_blank" rel="noreferrer">
                View on GitHub
              </a>
            </Button>
          </div>
        </div>

        {/* ── How it works ── */}
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-center text-muted-foreground uppercase tracking-widest text-sm">
            How it works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {STEPS.map((s) => (
              <div key={s.step} className="relative flex flex-col items-center text-center p-6 rounded-xl border border-border bg-card space-y-3">
                <span className="absolute top-4 right-4 text-4xl font-black text-muted/30 select-none tabular-nums">
                  {s.step}
                </span>
                <div className="p-3 rounded-lg bg-primary/10">
                  <s.icon className="w-5 h-5 text-primary" />
                </div>
                <h3 className="font-semibold">{s.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{s.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── Features ── */}
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-center text-muted-foreground uppercase tracking-widest text-sm">
            Five analysis modes
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="group flex flex-col gap-3 p-5 rounded-xl border border-border bg-card hover:border-primary/40 hover:shadow-sm transition-all cursor-default"
              >
                <div className={cn('w-9 h-9 rounded-lg flex items-center justify-center shrink-0', f.bg)}>
                  <f.icon className={cn('w-4 h-4', f.color)} />
                </div>
                <div className="space-y-1">
                  <h3 className="font-semibold text-sm">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.description}</p>
                </div>
              </div>
            ))}

            {/* Sources card */}
            <div className="flex flex-col gap-3 p-5 rounded-xl border border-border bg-card cursor-default">
              <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 bg-cyan-500/10">
                <FileSearch className="w-4 h-4 text-cyan-500" />
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-sm">Supported sources</h3>
                <div className="flex flex-wrap gap-1.5">
                  {['FIR', 'Court Order', 'Witness Statement', 'PDF', 'MP4 / Video', 'YouTube'].map(s => (
                    <Badge key={s} variant="secondary" className="text-xs">{s}</Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Tech stack ── */}
        <div className="space-y-4">
          <h2 className="text-sm font-semibold text-center text-muted-foreground uppercase tracking-widest">
            Built on
          </h2>
          <div className="flex flex-wrap justify-center gap-2">
            {TECH.map((t) => (
              <div
                key={t.label}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border bg-muted/50 text-sm text-muted-foreground"
              >
                <span>{t.icon}</span>
                <span className="font-medium">{t.label}</span>
              </div>
            ))}
          </div>
          <p className="text-center text-xs text-muted-foreground/60 pt-2">
            Multi-modal RAG · Cross-encoder reranking · LanceDB vector store · WhisperX transcription
          </p>
        </div>

        {/* ── CTA footer ── */}
        <div className="text-center pb-8 space-y-4">
          <Button size="lg" onClick={onCreateCase} className="gap-2 text-base px-8">
            <Upload className="w-4 h-4" />
            Get started — create a case
          </Button>
          <p className="text-xs text-muted-foreground">
            No setup required · Works offline · All data stays local
          </p>
        </div>

      </div>
    </div>
  );
}
