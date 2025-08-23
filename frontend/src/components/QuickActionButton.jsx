import React from 'react';

const QuickActionButton = ({ icon, children, onClick, className = '' }) => (
  <button onClick={onClick} className={className}>
    {icon}
    {children}
  </button>
);

export default QuickActionButton;
