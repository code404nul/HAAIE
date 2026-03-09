export type Emotion =
  | 'neutral' | 'approval' | 'gratitude' | 'curiosity' | 'amusement' | 'surprise'
  | 'joy' | 'excitement' | 'pride' | 'relief' | 'optimism' | 'desire' | 'realization' | 'caring'
  | 'sadness' | 'anger' | 'fear' | 'grief' | 'remorse' | 'disappointment' | 'confusion'
  | 'nervousness' | 'embarrassment' | 'love' | 'annoyance' | 'disgust' | 'disapproval';

export type Complexity = 'simple' | 'moderate' | 'complex';
export type Valence = 'positive' | 'negative' | 'neutral';

export interface EmotionEntry {
  timestamp: Date;
  emotion: Emotion;
  complexity: Complexity;
  valence: Valence;
}

export interface DayStats {
  date: string;
  positive: number;
  negative: number;
  neutral: number;
  total: number;
  dominantEmotion: Emotion;
}

export interface EmotionStats {
  entries: EmotionEntry[];
  totalCount: number;
  byEmotion: Record<string, number>;
  byValence: { positive: number; negative: number; neutral: number };
  byComplexity: { simple: number; moderate: number; complex: number };
  byDay: DayStats[];
  byHour: number[];
  topEmotions: Array<{ emotion: Emotion; count: number; valence: Valence }>;
  moodScore: number; // -100 to +100
  streak: { positive: number; negative: number };
}
