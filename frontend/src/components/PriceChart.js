import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { getMarketData } from '../services/api'
import { symbols } from '../config/symbols' // has 'EUR/USD', etc.

const toTs = (d) => new Date(d).getTime()

export default function PriceChart({ instrument='EUR/USD', hours=48 }) {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const now = Date.now()
    const from = now - hours*3600*1000
    ;(async () => {
      try {
        setLoading(true)
        setError(null)
        // Convert instrument to match backend format (EUR/USD -> EUR_USD)
        const backendInstrument = instrument.replace('/', '_')
        const res = await getMarketData(`?instrument=${backendInstrument}&from=${from}&to=${now}`)
        // Normalize to {time,label,close}
        const data = (res.rows || []).map(k => ({ t: k.t, c: k.c }))
        setRows(data)
      } catch (e) {
        setError('Failed to load market data')
      } finally {
        setLoading(false)
      }
    })()
  }, [instrument, hours])

  if (loading) return <div>Loading price…</div>
  if (error)   return <div className="text-red-500">{error}</div>

  return (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="t"
          domain={['auto','auto']}
          type="number"
          tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
        />
        <YAxis dataKey="c" domain={['auto','auto']} />
        <Tooltip
          labelFormatter={(ts) => new Date(ts).toLocaleString()}
          formatter={(v) => [v, 'Price']}
        />
        <Line type="monotone" dataKey="c" dot={false} strokeWidth={1.8} />
      </LineChart>
    </ResponsiveContainer>
  )
}