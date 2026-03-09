import { parseRawData, computeStats, EMOTION_EMOJI } from './emotions';
import {
  renderValenceDonut,
  renderDailyBar,
  renderTopEmotions,
  renderComplexityRadar,
  renderHourLine,
  renderMoodTrend,
} from './charts';
import type { EmotionStats } from './types';

// ── State ─────────────────────────────────────────────────────────────────────
let allStats: EmotionStats | null = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
function $(id: string): HTMLElement {
  return document.getElementById(id) as HTMLElement;
}

function fmt(n: number): string {
  return n.toLocaleString('fr-FR');
}

function pct(n: number, total: number): string {
  if (total === 0) return '0%';
  return ((n / total) * 100).toFixed(1) + '%';
}

function moodLabel(score: number): { text: string; color: string } {
  if (score >= 60) return { text: 'Très positif 🌟', color: '#4ade80' };
  if (score >= 20) return { text: 'Plutôt positif 😊', color: '#86efac' };
  if (score >= -20) return { text: 'Équilibré 😐', color: '#94a3b8' };
  if (score >= -60) return { text: 'Plutôt négatif 😔', color: '#fca5a5' };
  return { text: 'Très négatif 😞', color: '#f87171' };
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
function renderKPIs(stats: EmotionStats): void {
  const { totalCount, byValence, moodScore, streak } = stats;
  const mood = moodLabel(moodScore);

  $('kpi-total').textContent = fmt(totalCount);
  $('kpi-positive').textContent = fmt(byValence.positive);
  $('kpi-positive-pct').textContent = pct(byValence.positive, totalCount);
  $('kpi-negative').textContent = fmt(byValence.negative);
  $('kpi-negative-pct').textContent = pct(byValence.negative, totalCount);
  $('kpi-mood').textContent = `${moodScore > 0 ? '+' : ''}${moodScore}`;
  ($('kpi-mood') as HTMLElement).style.color = mood.color;
  $('kpi-mood-label').textContent = mood.text;

  const streakEl = $('kpi-streak');
  if (streak.positive > streak.negative) {
    streakEl.textContent = `${streak.positive}j positifs consécutifs`;
    streakEl.style.color = '#4ade80';
  } else if (streak.negative > 0) {
    streakEl.textContent = `${streak.negative}j négatifs consécutifs`;
    streakEl.style.color = '#f87171';
  } else {
    streakEl.textContent = 'Série mixte';
    streakEl.style.color = '#94a3b8';
  }

  // Mood bar
  const barFill = $('mood-bar-fill') as HTMLElement;
  const clampedScore = Math.max(-100, Math.min(100, moodScore));
  const pctPos = ((clampedScore + 100) / 200) * 100;
  barFill.style.width = pctPos + '%';
  barFill.style.background = mood.color;
}

// ── Timeline (last 10) ────────────────────────────────────────────────────────
function renderTimeline(stats: EmotionStats): void {
  const container = $('timeline');
  const last10 = [...stats.entries].slice(-20).reverse();
  container.innerHTML = last10.map(e => {
    const valColor = e.valence === 'positive' ? '#4ade80' : e.valence === 'negative' ? '#f87171' : '#94a3b8';
    const time = e.timestamp.toLocaleString('fr-FR', {
      day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'
    });
    return `
      <div class="timeline-item">
        <span class="tl-emoji">${EMOTION_EMOJI[e.emotion] ?? '●'}</span>
        <div class="tl-body">
          <span class="tl-name" style="color:${valColor}">${e.emotion}</span>
          <span class="tl-meta">${e.complexity} · ${time}</span>
        </div>
        <span class="tl-badge tl-${e.valence}">${e.valence}</span>
      </div>`;
  }).join('');
}

// ── Date range filter ─────────────────────────────────────────────────────────
function applyFilter(range: '7d' | '30d' | '90d' | 'all'): void {
  if (!allStats) return;
  const now = Date.now();
  const cutoff: Record<string, number> = { '7d': 7, '30d': 30, '90d': 90 };
  const entries = range === 'all'
    ? allStats.entries
    : allStats.entries.filter(e => {
        const age = (now - e.timestamp.getTime()) / 86400000;
        return age <= cutoff[range];
      });

  const filtered = computeStats(entries);
  renderKPIs(filtered);
  renderTimeline(filtered);
  renderValenceDonut(filtered);
  renderDailyBar(filtered);
  renderTopEmotions(filtered);
  renderComplexityRadar(filtered);
  renderHourLine(filtered);
  renderMoodTrend(filtered);
}

// ── Load JSON ─────────────────────────────────────────────────────────────────
async function loadData(): Promise<void> {
  try {
    const res = await fetch('/stats_user/feeling_history.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = await res.json();
    const entries = parseRawData(raw);
    allStats = computeStats(entries);

    $('status').textContent = `${fmt(entries.length)} entrées chargées`;
    $('status').style.color = '#4ade80';

    applyFilter('all');
    setupFilters();
  } catch (err) {
    console.error(err);
    $('status').textContent = 'Erreur de chargement — données de démo';
    $('status').style.color = '#f87171';
    loadDemo();
  }
}

// ── Demo fallback ─────────────────────────────────────────────────────────────
function loadDemo(): void {
  const emotions = [
    'gratitude','confusion','amusement','annoyance','joy','sadness',
    'excitement','fear','curiosity','disappointment','love','anger',
    'pride','nervousness','relief','disgust','optimism','remorse',
    'neutral','caring','approval','desire','surprise',
  ] as const;

  const raw = Array.from({ length: 400 }, (_, i) => {
    const d = new Date(Date.now() - Math.random() * 90 * 86400000);
    const emotion = emotions[Math.floor(Math.random() * emotions.length)];
    return { [d.toISOString().replace('T', ' ').slice(0, 26)]: emotion };
  });

  const entries = parseRawData(raw);
  allStats = computeStats(entries);
  applyFilter('all');
  setupFilters();
}

// ── Filter buttons ────────────────────────────────────────────────────────────
function setupFilters(): void {
  document.querySelectorAll<HTMLButtonElement>('[data-range]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-range]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilter(btn.dataset.range as '7d' | '30d' | '90d' | 'all');
    });
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadData();
});
