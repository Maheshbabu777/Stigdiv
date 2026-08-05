import React, { useRef, useEffect } from 'react';
import { ArrowUp } from 'lucide-react';

interface InputBoxProps {
  value: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export const InputBox: React.FC<InputBoxProps> = ({
  value,
  onChange,
  onSubmit,
  disabled,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !disabled) {
        onSubmit();
      }
    }
  };

  const hasText = value.trim().length > 0;

  return (
    <div className="input-box-wrapper">
      <div className="input-card">
        <textarea
          ref={textareaRef}
          className="user-textarea"
          placeholder="Ask anything about stocks, signals, or filings..."
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          autoFocus
        />
        <div className="input-actions">
          <button
            type="button"
            className={`send-button ${hasText && !disabled ? 'active' : ''}`}
            onClick={onSubmit}
            disabled={!hasText || disabled}
            aria-label="Send message"
          >
            <ArrowUp size={17} strokeWidth={2.4} />
          </button>
        </div>
      </div>
      <div className="disclaimer">
        Signal Divergence can make mistakes. Verify critical financial decisions.
      </div>
    </div>
  );
};

