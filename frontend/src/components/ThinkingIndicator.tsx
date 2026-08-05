import React from 'react';

export const ThinkingIndicator: React.FC = () => {
  return (
    <div className="message-row assistant">
      <div className="thinking-row">
        <div className="pulsing-dots">
          <span />
          <span />
          <span />
        </div>
        <span>Analyzing market signals...</span>
      </div>
    </div>
  );
};
