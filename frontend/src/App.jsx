import { useEffect, useState, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'

// ── WebSocket hook ─────────────────────────────────────────────
function useKairosWS(url) {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        // Reconnect after 2 seconds
        setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        try { setData(JSON.parse(e.data)) }
        catch { /* ignore parse errors */ }
      }
    }
    connect()
    return () => wsRef.current?.close()
  }, [url])

  return { data, connected }
}

// ── Metric card ────────────────────────────────────────────────
function MetricCard({ label, value, sub, color }) {
  return (
    <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
      <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1">
        {label}
      </p>
      <p className={`text-2xl font-bold ${color || 'text-white'}`}>{value}</p>
      {sub && <p className="text-slate-500 text-xs mt-1">{sub}</p>}
    </div>
  )
}

// ── Risk gauge ─────────────────────────────────────────────────
function RiskGauge({ drawdown, maxDrawdown = 0.02 }) {
  const pct = Math.min(Math.abs(drawdown) / maxDrawdown, 1)
  const color = pct < 0.5 ? '#22c55e' : pct < 0.8 ? '#f59e0b' : '#ef4444'
  return (
    <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
      <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-3">
        Risk utilization
      </p>
      <div className="w-full bg-slate-700 rounded-full h-3 mb-2">
        <div
          className="h-3 rounded-full transition-all duration-500"
          style={{ width: `${pct * 100}%`, backgroundColor: color }}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-500">
        <span>0%</span>
        <span style={{ color }}>
          {(Math.abs(drawdown) * 100).toFixed(2)}% of {(maxDrawdown * 100).toFixed(0)}% limit
        </span>
        <span>100%</span>
      </div>
    </div>
  )
}

// ── Agent signal panel ─────────────────────────────────────────
function AgentPanel({ agentState }) {
  const symbols = Object.keys(agentState || {})
  if (!symbols.length) {
    return (
      <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
        <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-3">
          Agent signals
        </p>
        <p className="text-slate-500 text-sm">Waiting for first cycle...</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
      <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-3">
        Agent signals
      </p>
      <div className="space-y-3">
        {symbols.map(sym => {
          const s = agentState[sym]
          const actionColor = {
            ENTER_LONG: 'text-green-400',
            EXIT_POSITION: 'text-red-400',
            HOLD: 'text-slate-400',
          }[s.action] || 'text-slate-400'

          return (
            <div key={sym} className="border-l-2 border-slate-600 pl-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-bold text-white">{sym}</span>
                <span className={`text-sm font-semibold ${actionColor}`}>
                  {s.action}
                </span>
                <span className="text-slate-500 text-xs ml-auto">
                  conf {((s.confidence || 0) * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-slate-500 text-xs">{s.reason}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Live price ticker ──────────────────────────────────────────
function StreamTicker({ stream }) {
  if (!stream || !Object.keys(stream).length) return null

  return (
    <div className="flex gap-4 mb-6">
      {Object.entries(stream).map(([ticker, bar]) => (
        <div key={ticker}
          className="bg-slate-800 rounded px-4 py-2 border border-slate-700 flex items-center gap-3">
          <span className="font-bold text-blue-400">{ticker}</span>
          <span className="text-white font-mono">${bar.close.toFixed(2)}</span>
          <span className="text-slate-500 text-xs">
            vol {(bar.volume / 1000).toFixed(0)}k
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Main app ───────────────────────────────────────────────────
export default function App() {
  const { data, connected } = useKairosWS('ws://localhost:8000/ws/live')

  if (!data) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="text-blue-400 text-4xl mb-4">⬡</div>
          <p className="text-slate-400">
            {connected ? 'Loading data...' : 'Connecting to Kairos...'}
          </p>
        </div>
      </div>
    )
  }

  const { portfolio, positions, orders, agent_state, stream, equity_curve } = data
  const pnlColor = (portfolio.daily_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
  const pnlSign = (portfolio.daily_pnl || 0) >= 0 ? '+' : ''

  // Format equity curve for recharts
  const curveData = (equity_curve || []).map((p, i) => ({
    i,
    equity: p.equity,
    t: new Date(p.t).toLocaleTimeString(),
  }))

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">

      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-extrabold text-blue-400 tracking-tight">
            ⬡ Kairos
          </h1>
          <p className="text-slate-500 text-sm">Autonomous Trading System</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
          <span className="text-slate-400 text-sm">
            {connected ? 'Live' : 'Reconnecting...'}
          </span>
        </div>
      </header>

      {/* Live price tickers */}
      <StreamTicker stream={stream} />

      {/* Portfolio metrics */}
      {/* Replace the existing 4-card grid with this 5-card version */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <MetricCard
          label="Equity"
          value={`$${(portfolio.equity || 0).toFixed(2)}`}
        />
        <MetricCard
          label="Cash"
          value={`$${(portfolio.cash || 0).toFixed(2)}`}
        />
        <MetricCard
          label="Daily P&L"
          value={`${pnlSign}$${(portfolio.daily_pnl || 0).toFixed(2)}`}
          color={pnlColor}
        />
        <MetricCard
          label="Total P&L"
          value={`$${(portfolio.total_pnl || 0).toFixed(2)}`}
          color={(portfolio.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <MetricCard
          label="Sharpe ratio"
          value={(data.sharpe || 0).toFixed(2)}
          color={
            (data.sharpe || 0) > 1 ? 'text-green-400' :
              (data.sharpe || 0) > 0 ? 'text-yellow-400' : 'text-red-400'
          }
          sub="annualised · live"
        />
      </div>

      {/* Equity curve + Risk gauge */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="md:col-span-2 bg-slate-800 rounded-lg p-5 border border-slate-700">
          <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-3">
            Equity curve
          </p>
          {curveData.length > 1 ? (
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={curveData}>
                <XAxis dataKey="t" tick={false} />
                <YAxis
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 11, fill: '#94a3b8' }}
                  width={70}
                  tickFormatter={v => `$${v.toFixed(0)}`}
                />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                  formatter={v => [`$${v.toFixed(2)}`, 'Equity']}
                  labelFormatter={l => l}
                />
                <ReferenceLine
                  y={10000}
                  stroke="#475569"
                  strokeDasharray="4 2"
                  label={{ value: 'Start', fill: '#64748b', fontSize: 10 }}
                />
                <Line
                  type="monotone"
                  dataKey="equity"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-40 flex items-center justify-center text-slate-600 text-sm">
              Waiting for portfolio snapshots...
            </div>
          )}
        </div>

        <RiskGauge
          drawdown={portfolio.drawdown_pct || 0}
          maxDrawdown={0.02}
        />
      </div>

      {/* Agent signals + Open positions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <AgentPanel agentState={agent_state} />

        <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
          <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-3">
            Open positions
          </p>
          {positions.length === 0 ? (
            <p className="text-slate-500 text-sm">No open positions.</p>
          ) : (
            <div className="space-y-2">
              {positions.map((p, i) => {
                const pnl = p.unrealized_pnl || 0
                return (
                  <div key={i}
                    className="flex items-center justify-between py-2 border-b border-slate-700 last:border-0">
                    <div>
                      <span className="font-bold">{p.ticker}</span>
                      <span className="text-slate-400 text-sm ml-2">
                        {p.quantity} shares @ ${p.avg_entry_price?.toFixed(2)}
                      </span>
                    </div>
                    <span className={pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Orders table */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        <div className="p-5 border-b border-slate-700">
          <h2 className="font-semibold">Recent executions</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="bg-slate-900 text-slate-400 text-xs uppercase">
                <th className="p-4">Time</th>
                <th className="p-4">Symbol</th>
                <th className="p-4">Side</th>
                <th className="p-4">Qty</th>
                <th className="p-4">Price</th>
                <th className="p-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.length === 0 ? (
                <tr>
                  <td colSpan={6} className="p-4 text-center text-slate-500">
                    No trades executed yet.
                  </td>
                </tr>
              ) : orders.map(o => (
                <tr key={o.id}
                  className="border-b border-slate-700 hover:bg-slate-700/40 transition-colors">
                  <td className="p-4 text-slate-400">
                    {new Date(o.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="p-4 font-bold">{o.ticker}</td>
                  <td className={`p-4 font-semibold ${o.side === 'buy' ? 'text-green-400' : 'text-red-400'
                    }`}>
                    {o.side?.toUpperCase()}
                  </td>
                  <td className="p-4">{o.qty}</td>
                  <td className="p-4">${o.price?.toFixed(2) || '—'}</td>
                  <td className="p-4">
                    <span className="bg-blue-900/40 text-blue-300 px-2 py-0.5 rounded text-xs">
                      {o.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}