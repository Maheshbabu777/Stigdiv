import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { StockChart } from './StockChart';
import { ChartData, Citation } from '../types';
import { ExternalLink, Database, Globe, MessageSquare } from 'lucide-react';

interface AssistantMessageProps {
  content: string;
  chartData?: ChartData | null;
  citations?: Citation[];
  isStreaming?: boolean;
}

export const AssistantMessage: React.FC<AssistantMessageProps> = ({
  content,
  chartData,
  citations,
  isStreaming,
}) => {
  // Helper to get matching icon for source
  const getSourceIcon = (source: string) => {
    const s = source.toLowerCase();
    if (s.includes('market') || s.includes('yfinance') || s.includes('price')) {
      return <Database size={11} className="citation-source-icon" />;
    }
    if (s.includes('social') || s.includes('reddit') || s.includes('stocktwits')) {
      return <MessageSquare size={11} className="citation-source-icon" />;
    }
    return <Globe size={11} className="citation-source-icon" />;
  };

  // Custom components for ReactMarkdown to handle links and inline citations cleanly
  const markdownComponents = {
    // Render links cleanly without long ugly URLs
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="citation-badge"
        title={href}
      >
        <ExternalLink size={9} className="citation-source-icon" />
        <span>{children}</span>
      </a>
    ),
  };

  return (
    <div className="message-row assistant">
      <div className="assistant-card">
        {/* Stock Chart Placed First */}
        {chartData && <StockChart data={chartData} />}

        {/* Markdown Streamed Content */}
        <div className={`prose ${isStreaming ? 'streaming-cursor' : ''}`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {content}
          </ReactMarkdown>
        </div>

        {/* Perplexity-style Minimal Sources Row */}
        {!isStreaming && citations && citations.length > 0 && (
          <div className="citations-footer">
            <span className="citations-label">Sources</span>
            {citations.map((c) =>
              c.url ? (
                <a
                  key={c.id}
                  href={c.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="citation-chip is-link"
                  title={`Open source: ${c.title}`}
                >
                  {getSourceIcon(c.source)}
                  <span className="citation-chip-text">{c.title || c.source}</span>
                  <ExternalLink size={10} className="citation-chip-ext" />
                </a>
              ) : (
                <span key={c.id} className="citation-chip" title={c.title}>
                  {getSourceIcon(c.source)}
                  <span className="citation-chip-text">{c.title || c.source}</span>
                </span>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
};
