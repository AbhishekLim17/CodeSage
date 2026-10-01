import React from 'react';

type Props = { label: string };

export default function Widget({ label }: Props) {
  return <button className="widget">{label}</button>;
}

export const Badge = ({ text }: { text: string }) => <span>{text}</span>;
