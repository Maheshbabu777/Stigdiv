import React from 'react';
import { Plus } from 'lucide-react';

interface TopBarProps {
  onNewChat: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ onNewChat }) => {
  return (
    <header className="top-bar">
      <button 
        className="new-chat-btn" 
        onClick={onNewChat} 
        aria-label="Start a new chat"
      >
        <Plus size={15} strokeWidth={2.2} />
        <span>New chat</span>
      </button>
    </header>
  );
};
