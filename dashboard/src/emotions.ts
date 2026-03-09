import type { Emotion, Complexity, Valence, EmotionEntry, EmotionStats, DayStats } from './types';

// ── Classification Maps ───────────────────────────────────────────────────────

export const COMPLEXITY_MAP: Record<Complexity, Emotion[]> = {
  simple: ['neutral', 'approval', 'gratitude', 'curiosity', 'amusement', 'surprise'],
  moderate: ['joy', 'excitement', 'pride', 'relief', 'optimism', 'desire', 'realization', 'caring'],
  complex: ['sadness', 'anger', 'fear', 'grief', 'remorse', 'disappointment', 'confusion',
            'nervousness', 'embarrassment', 'love', 'annoyance', 'disgust', 'disapproval'],
};

export const VALENCE_MAP: Record<Valence, Emotion[]> = {
  positive: ['gratitude', 'joy', 'amusement', 'approval', 'curiosity', 'excitement',
             'pride', 'relief', 'optimism', 'realization', 'caring', 'love', 'surprise'],
  neutral:  ['neutral'],
  negative: ['sadness', 'anger', 'fear', 'grief', 'remorse', 'disappointment', 'confusion',
             'nervousness', 'embarrassment', 'annoyance', 'disgust', 'disapproval', 'desire'],
};

export const EMOTION_EMOJI: Record<Emotion, string> = {
  neutral: '😐', approval: '👍', gratitude: '🙏', curiosity: '🔍', amusement: '😄',
  surprise: '😲', joy: '😊', excitement: '🎉', pride: '🦁', relief: '😮‍💨',
  optimism: '🌟', desire: '✨', realization: '💡', caring: '💚',
  sadness: '😔', anger: '😠', fear: '😨', grief: '💔', remorse: '😞',
  disappointment: '😕', confusion: '😵', nervousness: '😰', embarrassment: '😳',
  love: '❤️', annoyance: '😤', disgust: '🤢', disapproval: '👎',
};

// ── Classifier ────────────────────────────────────────────────────────────────

function getComplexity(emotion: Emotion): Complexity {
  for (const [complexity, emotions] of Object.entries(COMPLEXITY_MAP)) {
    if ((emotions as string[]).includes(emotion)) return complexity as Complexity;
  }
  return 'simple';
}

function getValence(emotion: Emotion): Valence {
  for (const [valence, emotions] of Object.entries(VALENCE_MAP)) {
    if ((emotions as string[]).includes(emotion)) return valence as Valence;
  }
  return 'neutral';
}

// ── Raw JSON Parser ───────────────────────────────────────────────────────────

type RawEntry = Record<string, string>;

export function parseRawData(raw: RawEntry[]): EmotionEntry[] {
  return raw.map((obj) => {
    const [timestampStr, emotionStr] = Object.entries(obj)[0];
    const emotion = emotionStr as Emotion;
    return {
      timestamp: new Date(timestampStr),
      emotion,
      complexity: getComplexity(emotion),
      valence: getValence(emotion),
    };
  }).filter(e => !isNaN(e.timestamp.getTime()))
    .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
}

// ── Aggregator ────────────────────────────────────────────────────────────────

export function computeStats(entries: EmotionEntry[]): EmotionStats {
  const byEmotion: Record<string, number> = {};
  const byValence = { positive: 0, negative: 0, neutral: 0 };
  const byComplexity = { simple: 0, moderate: 0, complex: 0 };
  const byHour: number[] = Array(24).fill(0);
  const dayMap: Map<string, { positive: number; negative: number; neutral: number; emotions: Record<string, number> }> = new Map();

  for (const e of entries) {
    byEmotion[e.emotion] = (byEmotion[e.emotion] ?? 0) + 1;
    byValence[e.valence]++;
    byComplexity[e.complexity]++;
    byHour[e.timestamp.getHours()]++;

    const dateKey = e.timestamp.toISOString().split('T')[0];
    if (!dayMap.has(dateKey)) dayMap.set(dateKey, { positive: 0, negative: 0, neutral: 0, emotions: {} });
    const day = dayMap.get(dateKey)!;
    day[e.valence]++;
    day.emotions[e.emotion] = (day.emotions[e.emotion] ?? 0) + 1;
  }

  const byDay: DayStats[] = Array.from(dayMap.entries()).map(([date, d]) => ({
    date,
    positive: d.positive,
    negative: d.negative,
    neutral: d.neutral,
    total: d.positive + d.negative + d.neutral,
    dominantEmotion: Object.entries(d.emotions).sort((a, b) => b[1] - a[1])[0]?.[0] as Emotion ?? 'neutral',
  })).sort((a, b) => a.date.localeCompare(b.date));

  const topEmotions = Object.entries(byEmotion)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([emotion, count]) => ({ emotion: emotion as Emotion, count, valence: getValence(emotion as Emotion) }));

  // Mood score: weighted average (positive = +1, neutral = 0, negative = -1) scaled to -100..+100
  const total = entries.length;
  const moodScore = total > 0
    ? Math.round(((byValence.positive - byValence.negative) / total) * 100)
    : 0;

  // Streak: count last consecutive days that were majority positive or negative
  let positiveStreak = 0;
  let negativeStreak = 0;
  for (let i = byDay.length - 1; i >= 0; i--) {
    const d = byDay[i];
    if (d.positive > d.negative) positiveStreak++;
    else break;
  }
  for (let i = byDay.length - 1; i >= 0; i--) {
    const d = byDay[i];
    if (d.negative > d.positive) negativeStreak++;
    else break;
  }

  return {
    entries,
    totalCount: entries.length,
    byEmotion,
    byValence,
    byComplexity,
    byDay,
    byHour,
    topEmotions,
    moodScore,
    streak: { positive: positiveStreak, negative: negativeStreak },
  };
}
