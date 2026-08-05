export interface ChartPoint {
  time: string;
  value: number;
}

export interface ChartData {
  ticker: string;
  period: string;
  points: ChartPoint[];
  latest_price?: number;
  pct_change?: number;
}

export interface Citation {
  id: number;
  source: string;
  title: string;
  url?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  chartData?: ChartData | null;
  citations?: Citation[];
  isStreaming?: boolean;
}
