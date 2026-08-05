import React, { useState, useEffect, useRef } from 'react';
import { TopBar } from './components/TopBar';
import { UserMessage } from './components/UserMessage';
import { AssistantMessage } from './components/AssistantMessage';
import { InputBox } from './components/InputBox';
import { ThinkingIndicator } from './components/ThinkingIndicator';
import { getRandomHeadline } from './data/headlines';
import { Message, ChartData, Citation } from './types';

export const App: React.FC = () => {
  const [headline, setHeadline] = useState<string>(getRandomHeadline());
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputVal, setInputVal] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>(() => {
    return localStorage.getItem('signal_session_id') || crypto.randomUUID();
  });

  const chatEndRef = useRef<HTMLDivElement>(null);
  const isZeroState = messages.length === 0;

  // Persist session id
  useEffect(() => {
    localStorage.setItem('signal_session_id', sessionId);
  }, [sessionId]);

  // Autoscroll when new messages appear
  useEffect(() => {
    if (!isZeroState) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading, isZeroState]);

  // Reset to clean zero state with a fresh session & random headline
  const handleNewChat = () => {
    const newId = crypto.randomUUID();
    setSessionId(newId);
    setMessages([]);
    setInputVal('');
    setIsLoading(false);
    setHeadline(getRandomHeadline());
  };

  const handleSendMessage = async () => {
    const query = inputVal.trim();
    if (!query || isLoading) return;

    // Add user message
    const userMsgId = crypto.randomUUID();
    const userMsg: Message = {
      id: userMsgId,
      role: 'user',
      content: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputVal('');
    setIsLoading(true);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          session_id: sessionId,
          use_llm: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setIsLoading(false);

      // Parse chart data if available
      let chartData: ChartData | null = null;
      if (data.chart_data && Array.isArray(data.chart_data.rows) && data.chart_data.rows.length > 0) {
        const rows = data.chart_data.rows;
        const points = rows
          .map((r: any) => ({
            time: r.date || r.time || '',
            value: typeof r.close === 'number' ? r.close : typeof r.value === 'number' ? r.value : 0,
          }))
          .filter((p: any) => p.time && p.value > 0);

        if (points.length > 0) {
          const firstVal = points[0].value;
          const lastVal = points[points.length - 1].value;
          const pct = firstVal !== 0 ? ((lastVal - firstVal) / firstVal) * 100 : 0;

          chartData = {
            ticker: data.chart_data.ticker || data.ticker || 'STOCK',
            period: data.chart_data.period || '5D',
            points,
            latest_price: lastVal,
            pct_change: pct,
          };
        }
      }

      // Generate Perplexity-style citations list from backend sources
      const citations: Citation[] = [];
      const currentTicker = data.chart_data?.ticker || data.ticker || '';
      if (data.sources) {
        // 1. Multi-source News & Institutional Media
        if (Array.isArray(data.sources.news)) {
          data.sources.news.slice(0, 8).forEach((item: any) => {
            if (item && item.title) {
              const publisher = item.publisher || item.source || 'Financial Media';
              const newsUrl =
                item.link ||
                item.url ||
                (currentTicker ? `https://finance.yahoo.com/quote/${currentTicker}/news` : undefined);
              citations.push({
                id: citations.length + 1,
                source: publisher,
                title: item.title,
                url: newsUrl,
              });
            }
          });
        }
        // 2. Real-time Market OHLCV & Fundamentals
        if (currentTicker && Array.isArray(data.sources.market) && data.sources.market.length > 0) {
          citations.push({
            id: citations.length + 1,
            source: 'Market OHLCV',
            title: `${currentTicker} Real-time Quotes & Fundamentals`,
            url: `https://finance.yahoo.com/quote/${currentTicker}`,
          });
        }
        // 3. Multi-channel Social & Retail Signals (StockTwits, Reddit, HN)
        if (Array.isArray(data.sources.social)) {
          data.sources.social.slice(0, 5).forEach((item: any) => {
            if (item && item.title) {
              const sourceLabel = item.source || item.publisher || 'Retail Sentiment';
              const socialUrl =
                item.link ||
                item.url ||
                (currentTicker ? `https://stocktwits.com/symbol/${currentTicker}` : undefined);
              citations.push({
                id: citations.length + 1,
                source: sourceLabel,
                title: item.title,
                url: socialUrl,
              });
            }
          });
        }
      }

      const fullText = data.response || 'No response received.';
      const assistantMsgId = crypto.randomUUID();

      // Initialize assistant message with chart rendered first and streaming text
      const assistantMsg: Message = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        chartData: chartData,
        citations: citations,
        isStreaming: true,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Stream text tokens with typewriter effect
      const words = fullText.split(' ');
      let currentWordIndex = 0;
      let accumulatedText = '';

      const streamInterval = setInterval(() => {
        if (currentWordIndex < words.length) {
          accumulatedText += (currentWordIndex === 0 ? '' : ' ') + words[currentWordIndex];
          currentWordIndex++;

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, content: accumulatedText }
                : msg
            )
          );
        } else {
          clearInterval(streamInterval);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, isStreaming: false }
                : msg
            )
          );
        }
      }, 20);
    } catch (err: any) {
      console.error('Chat error:', err);
      setIsLoading(false);
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error contacting intelligence engine: ${err.message || 'Unknown error'}. Please verify backend server is running.`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    }
  };

  return (
    <div className="app-container">
      {/* Sleek Top Navigation Bar */}
      <TopBar onNewChat={handleNewChat} />

      {isZeroState ? (
        /* Zero-State: Quirky Headline placed directly above the centered input card */
        <div className="zero-state-wrapper">
          <h1 className="zero-state-headline">{headline}</h1>
          <div className="zero-state-input-box">
            <InputBox
              value={inputVal}
              onChange={setInputVal}
              onSubmit={handleSendMessage}
              disabled={isLoading}
            />
          </div>
        </div>
      ) : (
        <>
          {/* Active Chat Feed */}
          <main className="main-content">
            <div className="chat-feed">
              {messages.map((msg) =>
                msg.role === 'user' ? (
                  <UserMessage key={msg.id} content={msg.content} />
                ) : (
                  <AssistantMessage
                    key={msg.id}
                    content={msg.content}
                    chartData={msg.chartData}
                    citations={msg.citations}
                    isStreaming={msg.isStreaming}
                  />
                )
              )}
              {isLoading && <ThinkingIndicator />}
              <div className="chat-bottom-spacer" />
              <div ref={chatEndRef} />
            </div>
          </main>

          {/* Fixed Bottom-Docked Input Box during Chat */}
          <div className="input-dock-fixed">
            <InputBox
              value={inputVal}
              onChange={setInputVal}
              onSubmit={handleSendMessage}
              disabled={isLoading}
            />
          </div>
        </>
      )}
    </div>
  );
};

export default App;
