import React from 'react';

interface UserMessageProps {
  content: string;
}

export const UserMessage: React.FC<UserMessageProps> = ({ content }) => {
  return (
    <div className="message-row user">
      <div className="user-bubble">
        {content}
      </div>
    </div>
  );
};
