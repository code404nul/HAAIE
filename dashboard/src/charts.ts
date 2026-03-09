import type { EmotionStats } from './types';

// Chart.js is loaded via CDN – declare global
declare const Chart: any;

const C = {
  pos: '#4ade80',
  neg: '#f87171',
  neu: '#94a3b8',
  simple: '#facc15',
  moderate: '#fb923c',
  complex: '#c084fc',
  bg: '#0f172a',
  surface: '#1e293b',
  border: '#334155',
  text: '#e2e8f0',
  muted: '#64748b',
};

Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 11;

const instances: Map<string, any> = new Map();

function destroy(id: string) {
  if (instances.has(id)) { instances.get(id).destroy(); instances.delete(id); }
}

function mk(id: string, config: object): void {
  destroy(id);
  const canvas = document.getElementById(id) as HTMLCanvasElement | null;
  if (!canvas) return;
  instances.set(id, new Chart(canvas, config));
}

// ── 1. Valence Donut ─────────────────────────────────────────────────────────
export function renderValenceDonut(stats: EmotionStats) {
  mk('chart-valence', {
    type: 'doughnut',
    data: {
      labels: ['Positif', 'Négatif', 'Neutre'],
      datasets: [{
        data: [stats.byValence.positive, stats.byValence.negative, stats.byValence.neutral],
        backgroundColor: [C.pos, C.neg, C.neu],
        borderWidth: 0,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 14, boxWidth: 10 } },
        tooltip: {
          callbacks: {
            label: (ctx: any) => {
              const pct = ((ctx.raw / stats.totalCount) * 100).toFixed(1);
              return ` ${ctx.label}: ${ctx.raw} (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

// ── 2. Daily Stacked Bar ──────────────────────────────────────────────────────
export function renderDailyBar(stats: EmotionStats) {
  const days = stats.byDay.slice(-30); // last 30 days
  const labels = days.map(d => {
    const [, m, day] = d.date.split('-');
    return `${day}/${m}`;
  });
  mk('chart-daily', {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Positif',
          data: days.map(d => d.positive),
          backgroundColor: C.pos + 'cc',
          borderRadius: 3, borderSkipped: false,
        },
        {
          label: 'Neutre',
          data: days.map(d => d.neutral),
          backgroundColor: C.neu + '88',
          borderRadius: 0, borderSkipped: false,
        },
        {
          label: 'Négatif',
          data: days.map(d => d.negative),
          backgroundColor: C.neg + 'cc',
          borderRadius: 3, borderSkipped: false,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 45 } },
        y: { stacked: true, grid: { color: C.border } }
      },
      plugins: { legend: { labels: { boxWidth: 10, padding: 14 } } }
    }
  });
}

// ── 3. Top Emotions Horizontal Bar ───────────────────────────────────────────
export function renderTopEmotions(stats: EmotionStats) {
  const top = stats.topEmotions;
  const colors = top.map(e =>
    e.valence === 'positive' ? C.pos : e.valence === 'negative' ? C.neg : C.neu
  );
  mk('chart-top', {
    type: 'bar',
    data: {
      labels: top.map(e => e.emotion),
      datasets: [{
        data: top.map(e => e.count),
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor: colors,
        borderWidth: 1,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: C.border } },
        y: { grid: { display: false } }
      }
    }
  });
}

// ── 4. Complexity Radar ───────────────────────────────────────────────────────
export function renderComplexityRadar(stats: EmotionStats) {
  const { simple, moderate, complex } = stats.byComplexity;
  const total = stats.totalCount || 1;
  mk('chart-complexity', {
    type: 'polarArea',
    data: {
      labels: ['Simple', 'Modérée', 'Complexe'],
      datasets: [{
        data: [simple, moderate, complex],
        backgroundColor: [C.simple + '99', C.moderate + '99', C.complex + '99'],
        borderColor: [C.simple, C.moderate, C.complex],
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: C.border },
          ticks: { display: false },
          pointLabels: { color: C.muted }
        }
      },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, padding: 12 } },
        tooltip: {
          callbacks: {
            label: (ctx: any) => ` ${ctx.raw} (${((ctx.raw / total) * 100).toFixed(1)}%)`
          }
        }
      }
    }
  });
}

// ── 5. Hour Heatmap (line) ────────────────────────────────────────────────────
export function renderHourLine(stats: EmotionStats) {
  mk('chart-hour', {
    type: 'line',
    data: {
      labels: Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}h`),
      datasets: [{
        label: 'Émotions',
        data: stats.byHour,
        borderColor: '#818cf8',
        backgroundColor: 'rgba(129,140,248,0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#818cf8',
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: C.border }, beginAtZero: true }
      }
    }
  });
}

// ── 6. Mood Trend Line ────────────────────────────────────────────────────────
export function renderMoodTrend(stats: EmotionStats) {
  const days = stats.byDay;
  const labels = days.map(d => d.date.slice(5)); // MM-DD
  const scores = days.map(d => {
    const t = d.total || 1;
    return Math.round(((d.positive - d.negative) / t) * 100);
  });

  mk('chart-mood', {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Score humeur',
        data: scores,
        borderColor: scores.map(s => s >= 0 ? C.pos : C.neg),
        segment: {
          borderColor: (ctx: any) => ctx.p1.parsed.y >= 0 ? C.pos : C.neg,
          backgroundColor: (ctx: any) => ctx.p1.parsed.y >= 0 ? C.pos + '22' : C.neg + '22',
        },
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        borderWidth: 2,
        pointBackgroundColor: scores.map(s => s >= 0 ? C.pos : C.neg),
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: {
          grid: { color: C.border },
          min: -100, max: 100,
          ticks: {
            callback: (v: number) => v > 0 ? `+${v}` : `${v}`
          }
        }
      }
    }
  });
}
