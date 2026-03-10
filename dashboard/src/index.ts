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
    await loadDependency();
    setupFilters();
  } catch (err) {
    console.error(err);
    $('status').textContent = 'Erreur de chargement — données de démo';
    $('status').style.color = '#f87171';
    loadDemo();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL DEPENDENCY MODULE
// ══════════════════════════════════════════════════════════════════════════════

interface DepEntry {
  timestamp: Date;
  score: number;
  niveau: string;
  indicateurs: string[];
  explication: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function depColor(score: number): string {
  if (score <= 20) return '#4ade80';
  if (score <= 50) return '#facc15';
  if (score <= 70) return '#fb923c';
  return '#f87171';
}

function depBadgeClass(niveau: string): string {
  if (niveau.toLowerCase().includes('sain')) return 'badge-pos';
  if (niveau.toLowerCase().includes('légèrement')) return 'badge-neu';
  return 'badge-neg';
}

function depEmoji(score: number): string {
  if (score === 0) return '💚';
  if (score <= 30) return '🟡';
  if (score <= 60) return '🟠';
  if (score <= 80) return '🔴';
  return '🚨';
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
function renderDepKPIs(entries: DepEntry[]): void {
  if (!entries.length) return;
  const latest = entries[entries.length - 1];
  const scores = entries.map(e => e.score);
  const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
  const maxScore = Math.max(...scores);
  const maxEntry = entries.find(e => e.score === maxScore)!;
  const critical = entries.filter(e => e.score >= 70).length;

  // Score actuel
  const scoreEl = $('dep-score');
  scoreEl.textContent = String(latest.score);
  scoreEl.style.color = depColor(latest.score);
  $('dep-niveau').textContent = latest.niveau;
  const badge = $('dep-niveau-badge');
  badge.textContent = latest.niveau;
  badge.className = 'card-badge ' + depBadgeClass(latest.niveau);
  const fill = $('dep-bar-fill') as HTMLElement;
  fill.style.left = latest.score + '%';
  fill.style.borderColor = depColor(latest.score);

  // Moyenne
  const avgEl = $('dep-avg');
  avgEl.textContent = String(avg);
  avgEl.style.color = depColor(avg);

  // Max
  const maxEl = $('dep-max');
  maxEl.textContent = String(maxScore);
  $('dep-max-date').textContent = maxEntry.timestamp.toLocaleDateString('fr-FR', {
    day: '2-digit', month: 'long', year: 'numeric'
  });

  // Critiques
  const critEl = $('dep-critical');
  critEl.textContent = String(critical);
  $('dep-critical-pct').textContent = pct(critical, entries.length) + ' des analyses';

  // Total badge
  $('dep-total-badge').textContent = `${entries.length} entrées`;
}

// ── Trend Chart ───────────────────────────────────────────────────────────────
function renderDepTrend(entries: DepEntry[]): void {
  const ctx = (document.getElementById('chart-dep-trend') as HTMLCanvasElement).getContext('2d')!;
  const labels = entries.map(e =>
    e.timestamp.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' })
  );
  const scores = entries.map(e => e.score);
  const C = (window as any).Chart;

  new C(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Score',
          data: scores,
          borderColor: '#818cf8',
          backgroundColor: (ctx2: any) => {
            const chart = ctx2.chart;
            const { ctx: c, chartArea } = chart;
            if (!chartArea) return 'rgba(129,140,248,.1)';
            const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            gradient.addColorStop(0, 'rgba(248,113,113,.25)');
            gradient.addColorStop(0.5, 'rgba(250,204,21,.1)');
            gradient.addColorStop(1, 'rgba(74,222,128,.05)');
            return gradient;
          },
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: scores.map(s => depColor(s)),
          pointBorderColor: 'var(--bg)',
          pointBorderWidth: 2,
          fill: true,
          tension: 0.4,
        },
        // Threshold line at 70 (malsain)
        {
          label: 'Seuil malsain',
          data: new Array(scores.length).fill(70),
          borderColor: 'rgba(248,113,113,.4)',
          borderDash: [4, 4],
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
        // Threshold line at 20 (sain)
        {
          label: 'Seuil sain',
          data: new Array(scores.length).fill(20),
          borderColor: 'rgba(74,222,128,.3)',
          borderDash: [4, 4],
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f1520',
          borderColor: '#1d2d44',
          borderWidth: 1,
          titleColor: '#94a3b8',
          bodyColor: '#e2e8f0',
          callbacks: {
            label: (item: any) => {
              if (item.datasetIndex !== 0) return null;
              const e = entries[item.dataIndex];
              const lines = [`Score : ${e.score} — ${e.niveau}`];
              if (e.indicateurs.length) lines.push(`Indicateurs : ${e.indicateurs.join(', ')}`);
              return lines;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#64748b', font: { size: 10, family: 'DM Mono' }, maxTicksLimit: 12 },
          grid: { color: 'rgba(255,255,255,.03)' }
        },
        y: {
          min: 0, max: 100,
          ticks: { color: '#64748b', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,.04)' }
        }
      }
    }
  });
}

// ── Niveau Donut ──────────────────────────────────────────────────────────────
function renderDepNiveau(entries: DepEntry[]): void {
  const counts: Record<string, number> = {};
  entries.forEach(e => { counts[e.niveau] = (counts[e.niveau] ?? 0) + 1; });
  const labels = Object.keys(counts);
  const data = labels.map(l => counts[l]);
  const colors = labels.map(l => {
    if (l.toLowerCase().includes('sain') && !l.toLowerCase().includes('malsain')) return '#4ade80';
    if (l.toLowerCase().includes('légèrement')) return '#facc15';
    if (l.toLowerCase().includes('modérément')) return '#fb923c';
    return '#f87171';
  });

  const ctx = (document.getElementById('chart-dep-niveau') as HTMLCanvasElement).getContext('2d')!;
  const C = (window as any).Chart;
  new C(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderColor: '#080c14', borderWidth: 3, hoverOffset: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#94a3b8', font: { size: 11, family: 'DM Mono' }, boxWidth: 10, padding: 12 }
        },
        tooltip: {
          backgroundColor: '#0f1520', borderColor: '#1d2d44', borderWidth: 1,
          titleColor: '#94a3b8', bodyColor: '#e2e8f0',
          callbacks: {
            label: (item: any) => ` ${item.label} : ${item.raw} (${pct(item.raw, entries.length)})`
          }
        }
      }
    }
  });
}

// ── Indicators Bar Chart ──────────────────────────────────────────────────────
function renderDepIndicators(entries: DepEntry[]): void {
  const counts: Record<string, number> = {};
  entries.forEach(e => e.indicateurs.forEach(ind => {
    counts[ind] = (counts[ind] ?? 0) + 1;
  }));
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (!sorted.length) return;

  const ctx = (document.getElementById('chart-dep-indicators') as HTMLCanvasElement).getContext('2d')!;
  const C = (window as any).Chart;
  new C(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(([k]) => k),
      datasets: [{
        data: sorted.map(([, v]) => v),
        backgroundColor: sorted.map(([, v]) => {
          const max = sorted[0][1];
          const ratio = v / max;
          if (ratio > 0.7) return 'rgba(248,113,113,.7)';
          if (ratio > 0.4) return 'rgba(251,146,60,.6)';
          return 'rgba(250,204,21,.5)';
        }),
        borderColor: 'transparent',
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { backgroundColor: '#0f1520', borderColor: '#1d2d44', borderWidth: 1, titleColor: '#94a3b8', bodyColor: '#e2e8f0' }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,.04)' } },
        y: { ticks: { color: '#94a3b8', font: { size: 10, family: 'DM Mono' } }, grid: { display: false } }
      }
    }
  });
}

// ── Hourly Score Chart ────────────────────────────────────────────────────────
function renderDepHour(entries: DepEntry[]): void {
  const hourSums: number[] = new Array(24).fill(0);
  const hourCounts: number[] = new Array(24).fill(0);
  entries.forEach(e => {
    const h = e.timestamp.getHours();
    hourSums[h] += e.score;
    hourCounts[h]++;
  });
  const avgs = hourSums.map((s, i) => hourCounts[i] > 0 ? Math.round(s / hourCounts[i]) : null);

  const ctx = (document.getElementById('chart-dep-hour') as HTMLCanvasElement).getContext('2d')!;
  const C = (window as any).Chart;
  new C(ctx, {
    type: 'bar',
    data: {
      labels: Array.from({ length: 24 }, (_, i) => `${i}h`),
      datasets: [{
        data: avgs,
        backgroundColor: avgs.map(v => v === null ? 'transparent' : depColor(v) + 'aa'),
        borderColor: avgs.map(v => v === null ? 'transparent' : depColor(v ?? 0)),
        borderWidth: 1,
        borderRadius: 3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { backgroundColor: '#0f1520', borderColor: '#1d2d44', borderWidth: 1, titleColor: '#94a3b8', bodyColor: '#e2e8f0',
          callbacks: { label: (item: any) => item.raw !== null ? ` Score moy. : ${item.raw}` : ' Aucune donnée' }
        }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { display: false } },
        y: { min: 0, max: 100, ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,.04)' } }
      }
    }
  });
}

// ── Latest Indicators + Explication ──────────────────────────────────────────
function renderDepLatestDetail(entries: DepEntry[]): void {
  const latest = entries[entries.length - 1];
  const indEl = $('dep-indicators');
  $('dep-ind-count').textContent = latest.indicateurs.length
    ? `${latest.indicateurs.length} indicateur${latest.indicateurs.length > 1 ? 's' : ''}`
    : 'aucun';

  if (!latest.indicateurs.length) {
    indEl.innerHTML = '<span style="font-size:.75rem;color:var(--pos)">✓ Aucun indicateur — état sain</span>';
  } else {
    // Count each indicator across all entries for context
    const allCounts: Record<string, number> = {};
    entries.forEach(e => e.indicateurs.forEach(ind => { allCounts[ind] = (allCounts[ind] ?? 0) + 1; }));
    indEl.innerHTML = latest.indicateurs.map(ind => {
      const freq = allCounts[ind] ?? 1;
      const freqPct = Math.round((freq / entries.length) * 100);
      return `<span style="font-size:.65rem;padding:.25rem .7rem;border-radius:4px;background:rgba(248,113,113,.1);color:var(--neg);border:1px solid rgba(248,113,113,.2);cursor:default" title="Apparu ${freq}× — ${freqPct}% des analyses">${ind} <span style="opacity:.5">${freqPct}%</span></span>`;
    }).join('');
  }
  $('dep-explication').textContent = latest.explication;
}

// ── Niveau Breakdown ─────────────────────────────────────────────────────────
function renderDepNiveauBreakdown(entries: DepEntry[]): void {
  const breakdown: Record<string, { count: number; totalScore: number }> = {};
  entries.forEach(e => {
    if (!breakdown[e.niveau]) breakdown[e.niveau] = { count: 0, totalScore: 0 };
    breakdown[e.niveau].count++;
    breakdown[e.niveau].totalScore += e.score;
  });

  const sorted = Object.entries(breakdown).sort((a, b) => b[1].totalScore / b[1].count - a[1].totalScore / a[1].count);
  const container = $('dep-niveau-breakdown');
  container.innerHTML = sorted.map(([niveau, data]) => {
    const avg = Math.round(data.totalScore / data.count);
    const share = (data.count / entries.length) * 100;
    const color = depColor(avg);
    return `
      <div style="display:flex;flex-direction:column;gap:.3rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:.72rem;color:${color}">${niveau}</span>
          <span style="font-size:.65rem;color:var(--muted)">${data.count}× · moy. ${avg}</span>
        </div>
        <div style="height:5px;background:var(--border);border-radius:100px;overflow:hidden">
          <div style="height:100%;width:${share}%;background:${color};border-radius:100px;transition:width .8s ease"></div>
        </div>
        <div style="font-size:.6rem;color:var(--muted)">${share.toFixed(1)}% des analyses</div>
      </div>`;
  }).join('');
}

// ── Full Timeline ─────────────────────────────────────────────────────────────
function renderDepTimeline(entries: DepEntry[]): void {
  const tlEl = $('dep-timeline');
  tlEl.innerHTML = [...entries].reverse().map(e => {
    const color = depColor(e.score);
    const badgeCls = depBadgeClass(e.niveau);
    const time = e.timestamp.toLocaleString('fr-FR', {
      day: '2-digit', month: '2-digit', year: '2-digit',
      hour: '2-digit', minute: '2-digit'
    });
    const indStr = e.indicateurs.length
      ? e.indicateurs.join(' · ')
      : 'Aucun indicateur';
    return `
      <div class="timeline-item" style="flex-direction:column;align-items:flex-start;gap:.4rem">
        <div style="display:flex;align-items:center;gap:.75rem;width:100%">
          <span style="font-size:1.1rem;flex-shrink:0">${depEmoji(e.score)}</span>
          <div style="flex:1;min-width:0">
            <span style="display:block;font-size:.8rem;font-weight:500;color:${color}">${e.niveau}</span>
            <span style="font-size:.63rem;color:var(--muted)">${time}</span>
          </div>
          <span class="tl-badge ${badgeCls}" style="font-size:.72rem;padding:.25rem .6rem">${e.score}/100</span>
        </div>
        ${e.indicateurs.length ? `
          <div style="display:flex;flex-wrap:wrap;gap:.3rem;padding-left:1.75rem">
            ${e.indicateurs.map(ind =>
              `<span style="font-size:.6rem;padding:.15rem .45rem;border-radius:3px;background:rgba(248,113,113,.08);color:var(--neg);border:1px solid rgba(248,113,113,.15)">${ind}</span>`
            ).join('')}
          </div>` : ''}
        <div style="padding-left:1.75rem;font-size:.65rem;color:var(--muted);font-style:italic;line-height:1.6">${e.explication}</div>
      </div>`;
  }).join('');
}

// ── Master render ─────────────────────────────────────────────────────────────
function renderDependency(entries: DepEntry[]): void {
  if (!entries.length) return;
  renderDepKPIs(entries);
  renderDepTrend(entries);
  renderDepNiveau(entries);
  renderDepIndicators(entries);
  renderDepHour(entries);
  renderDepLatestDetail(entries);
  renderDepNiveauBreakdown(entries);
  renderDepTimeline(entries);
}

// ── Loader ────────────────────────────────────────────────────────────────────
async function loadDependency(): Promise<void> {
  try {
    const res = await fetch('/stats_user/emmotionnal_dependency.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw: Array<{
      timestamp: string; score: number; niveau: string;
      indicateurs: string[]; explication: string;
    }> = await res.json();
    const entries: DepEntry[] = raw
      .map(r => ({
        timestamp: new Date(r.timestamp),
        score: r.score,
        niveau: r.niveau,
        indicateurs: r.indicateurs,
        explication: r.explication,
      }))
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
    renderDependency(entries);
  } catch (err) {
    console.warn('Dependency data unavailable:', err);
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
