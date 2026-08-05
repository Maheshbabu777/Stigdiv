import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { ChartData } from '../types';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StockChartProps {
  data: ChartData;
}

export const StockChart: React.FC<StockChartProps> = ({ data }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  const isPositive = (data.pct_change ?? 0) >= 0;
  const strokeColor = isPositive ? '#10B981' : '#EF4444';
  const topColor = isPositive ? 'rgba(16, 185, 129, 0.22)' : 'rgba(239, 68, 68, 0.22)';
  const bottomColor = 'rgba(255, 255, 255, 0.0)';

  useEffect(() => {
    if (!containerRef.current || !data.points || data.points.length === 0) return;

    // Clean up previous instance
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 270,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#8F8D85',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(0, 0, 0, 0.04)' },
        horzLines: { color: 'rgba(0, 0, 0, 0.04)' },
      },
      crosshair: {
        vertLine: {
          color: '#C96442',
          width: 1,
          style: 3,
          labelBackgroundColor: '#C96442',
        },
        horzLine: {
          color: '#C96442',
          width: 1,
          style: 3,
          labelBackgroundColor: '#C96442',
        },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: false,
      handleScale: false,
    });

    const areaSeries = chart.addAreaSeries({
      lineColor: strokeColor,
      topColor: topColor,
      bottomColor: bottomColor,
      lineWidth: 2,
      priceLineVisible: false,
    });

    // Format data points for lightweight-charts
    const formattedData = data.points
      .map((p) => {
        // lightweight-charts expects YYYY-MM-DD or Unix timestamp
        const timeStr = p.time.includes('T') ? p.time.split('T')[0] : p.time;
        return {
          time: timeStr,
          value: p.value,
        };
      })
      .filter((p, idx, arr) => idx === 0 || p.time !== arr[idx - 1].time) // deduplicate times
      .sort((a, b) => (a.time > b.time ? 1 : -1));

    if (formattedData.length > 0) {
      areaSeries.setData(formattedData);
      chart.timeScale().fitContent();
    }

    chartInstanceRef.current = chart;
    seriesRef.current = areaSeries;

    // Remove any branding/logo DOM elements added by the library
    const removeBranding = () => {
      if (containerRef.current) {
        const links = containerRef.current.querySelectorAll('a');
        links.forEach((el) => el.remove());
      }
    };
    removeBranding();

    const observer = new MutationObserver(removeBranding);
    observer.observe(containerRef.current, { childList: true, subtree: true });

    // Handle container resize
    const handleResize = () => {
      if (containerRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', handleResize);
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
        chartInstanceRef.current = null;
      }
    };
  }, [data, strokeColor, topColor]);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div className="chart-title-group">
          <span className="chart-ticker">{data.ticker}</span>
          <span className="chart-period-badge">{data.period.toUpperCase()} TIMELINE</span>
        </div>
        <div className="chart-price-group">
          {data.latest_price !== undefined && (
            <span className="chart-price">${data.latest_price.toFixed(2)}</span>
          )}
          {data.pct_change !== undefined && (
            <span className={`chart-pill ${isPositive ? 'positive' : 'negative'}`}>
              {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
              {isPositive ? '+' : ''}
              {data.pct_change.toFixed(2)}%
            </span>
          )}
        </div>
      </div>
      <div ref={containerRef} className="chart-canvas-wrapper" />
    </div>
  );
};
