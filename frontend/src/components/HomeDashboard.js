import React, { useState } from 'react'
import PriceChart from './PriceChart'
import { symbols } from '../config/symbols'

export default function HomeDashboard() {
  const [pair, setPair] = useState('EUR/USD') // default
  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center gap-2">
        <h2 className="text-xl font-semibold">Market</h2>
        <select
          className="border rounded px-2 py-1 text-sm"
          value={pair}
          onChange={(e) => setPair(e.target.value)}
          disabled // enable later when other pairs seeded
        >
          {symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <PriceChart instrument={pair} hours={48} />
    </div>
  )
}