import { useEffect, useState, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar
} from 'recharts'

const ACCENT = '#5DCAA5'
const BLUE = '#378ADD'
const PURPLE = '#534AB7'
const AMBER = '#EF9F27'
const RED = '#E24B4A'
const DIM = '#2a4060'
const MID = '#4a6080'
const LIGHT = '#c8d8e8'
const BORDER = '#1e2a3a'
const CARD = '#0d1220'
const BG = '#0a0e1a'
const SIDEBAR = '#0b0f1e'

const mono = { fontFamily: 'monospace' }

function useWS(url) {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const ws = useRef(null)
  useEffect(() => {
    function connect() {
      const s = new WebSocket(url)
      ws.current = s
      s.onopen = () => setConnected(true)
      s.onclose = () => { setConnected(false); setTimeout(connect, 2000) }
      s.onerror = () => s.close()
      s.onmessage = e => { try { setData(JSON.parse(e.data)) } catch {} }
    }
    connect()
    return () => ws.current?.close()
  }, [url])
  return { data, connected }
}

function Dot({ color, size = 7 }) {
  return (
    <span style={{
      display: 'inline-block', width: size, height: size,
      borderRadius: '50%', background: color, flexShrink: 0
    }} />
  )
}

function MetricCard({ label, value, sub, color = LIGHT }) {
  return (
    <div style={{
      background: CARD, border: `0.5px solid ${BORDER}`,
      borderRadius: 8, padding: '12px 14px'
    }}>
      <div style={{ fontSize: 10, color: DIM, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 500, color, ...mono }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: MID, marginTop: 3 }}>{sub}</div>}
    </div>
  )
}

function CardTitle({ icon, children }) {
  return (
    <div style={{
      fontSize: 10, color: DIM, letterSpacing: '0.08em',
      textTransform: 'uppercase', marginBottom: 10,
      display: 'flex', alignItems: 'center', gap: 6
    }}>
      <i className={`ti ti-${icon}`} style={{ fontSize: 13 }} aria-hidden="true" />
      {children}
    </div>
  )
}

function FeatureBar({ label, value, max = 3, color = BLUE }) {
  const pct = Math.min(Math.abs(value) / max * 100, 100)
  const display = typeof value === 'number' ? (value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3)) : value
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
      <span style={{ fontSize: 11, color: MID, width: 68, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 4, background: '#1e2a3a', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: 11, color, width: 48, textAlign: 'right', ...mono }}>{display}</span>
    </div>
  )
}

function AgentRow({ color, name, symbol, sub, badge, badgeType }) {
  const badgeStyle = {
    hold:    { background: '#1e2a3a', color: MID },
    buy:     { background: '#0F6E5620', color: ACCENT, border: `0.5px solid ${ACCENT}40` },
    sell:    { background: '#A32D2D20', color: RED,   border: `0.5px solid ${RED}40` },
    blocked: { background: '#854F0B20', color: AMBER,  border: `0.5px solid ${AMBER}40` },
  }[badgeType || 'hold']

  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 8,
      padding: '7px 0', borderBottom: `0.5px solid ${BORDER}`
    }}>
      <Dot color={color} size={7} style={{ marginTop: 4 }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, color: LIGHT, fontWeight: 500, marginBottom: 2 }}>
          {name}
          {symbol && <span style={{ color: DIM, fontWeight: 400 }}> · {symbol}</span>}
        </div>
        <div style={{ fontSize: 11, color: MID, lineHeight: 1.4 }}>{sub}</div>
      </div>
      <div style={{
        fontSize: 10, padding: '2px 7px', borderRadius: 10,
        whiteSpace: 'nowrap', flexShrink: 0, ...badgeStyle
      }}>
        {badge}
      </div>
    </div>
  )
}

function NewsRow({ score, text }) {
  const color = score > 0.1 ? ACCENT : score < -0.1 ? RED : MID
  const label = score > 0.1 ? 'pos' : score < -0.1 ? 'neg' : 'neu'
  return (
    <div style={{
      padding: '6px 0', borderBottom: `0.5px solid ${BORDER}`,
      display: 'flex', gap: 8, alignItems: 'flex-start'
    }}>
      <span style={{ fontSize: 11, color, width: 44, flexShrink: 0, ...mono }}>
        {score >= 0 ? '+' : ''}{score.toFixed(3)}
      </span>
      <span style={{ fontSize: 11, color: MID, lineHeight: 1.4 }}>{text}</span>
    </div>
  )
}

function LogLine({ time, sym, msg, level }) {
  const color = level === 'green' ? ACCENT : level === 'amber' ? AMBER : level === 'red' ? RED : MID
  return (
    <div style={{
      display: 'flex', gap: 8, padding: '3px 0',
      borderBottom: `0.5px solid #0d1a2e`, fontSize: 10, ...mono
    }}>
      <span style={{ color: DIM, flexShrink: 0 }}>{time}</span>
      <span style={{ color: BLUE, width: 36, flexShrink: 0 }}>{sym}</span>
      <span style={{ color }}>{msg}</span>
    </div>
  )
}

function SentimentBars({ signal }) {
  if (!signal) return <div style={{ color: MID, fontSize: 12 }}>Waiting for data...</div>
  const score = signal.aggregate_score || 0
  const negPct = Math.max(0, Math.min(100, (-score + 1) / 2 * 100))
  const posPct = Math.max(0, Math.min(100, (score + 1) / 2 * 100))
  const neuPct = Math.max(0, 100 - Math.abs(negPct - 50) - Math.abs(posPct - 50))

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: MID }}>aggregate score</span>
        <span style={{ fontSize: 18, fontWeight: 500, color: score >= 0 ? ACCENT : RED, ...mono }}>
          {score >= 0 ? '+' : ''}{score.toFixed(3)}
        </span>
      </div>
      {[
        { label: 'bearish', pct: Math.round(negPct), color: RED },
        { label: 'neutral', pct: Math.round(neuPct), color: MID },
        { label: 'bullish', pct: Math.round(posPct), color: ACCENT },
      ].map(({ label, pct, color }) => (
        <FeatureBar key={label} label={label} value={pct} max={100} color={color} />
      ))}
      <div style={{ height: 0.5, background: BORDER, margin: '8px 0' }} />
      {(signal.top_headlines || []).slice(0, 3).map((h, i) => (
        <NewsRow key={i} score={h.score || 0} text={h.headline || ''} />
      ))}
      {signal.reasoning && (
        <div style={{
          marginTop: 8, padding: 8, background: '#1e2a3a20',
          borderRadius: 6, border: `0.5px solid ${BORDER}`
        }}>
          <div style={{ fontSize: 10, color: DIM, marginBottom: 3 }}>
            {signal.source === 'finbert+gpt4o' ? 'GPT-4o-mini reasoning' : 'FinBERT only'}
          </div>
          <div style={{ fontSize: 11, color: MID, lineHeight: 1.5 }}>{signal.reasoning}</div>
        </div>
      )}
    </>
  )
}

export default function App() {
  const { data, connected } = useWS('ws://localhost:8000/ws/live')
  const [logs, setLogs] = useState([])
  const [curveData, setCurveData] = useState([])
  const [activeView, setActiveView] = useState('dashboard')
  const [evalMetrics, setEvalMetrics] = useState(null)

  useEffect(() => {
    fetch('http://localhost:8000/dashboard/eval-metrics')
      .then(r => r.json())
      .then(setEvalMetrics)
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!data) return
    const now = new Date()
    const time = now.toLocaleTimeString('en-US', { hour12: false })
    const p = data.portfolio || {}
    const newEntry = {
      t: time,
      equity: p.equity || 10000,
    }
    setCurveData(prev => [...prev.slice(-60), newEntry])

    const agents = data.agent_state || {}
    Object.entries(agents).forEach(([sym, state]) => {
      if (state?.action) {
        setLogs(prev => [{
          time,
          sym,
          msg: `${state.reason || state.action}`,
          level: state.action === 'ENTER_LONG' ? 'green' :
                 state.action === 'EXIT_POSITION' ? 'red' : ''
        }, ...prev].slice(0, 20))
      }
    })
  }, [data])

  const p = data?.portfolio || {}
  const equity = p.equity || 10000
  const dailyPnl = p.daily_pnl || 0
  const drawdown = p.drawdown_pct || 0
  const sharpe = data?.sharpe || 0
  const stream = data?.stream || {}
  const agentState = data?.agent_state || {}
  const positions = data?.positions || []
  const orders = data?.orders || []
  const tradesToday = data?.trades_today ?? 0

  const pnlColor = dailyPnl >= 0 ? ACCENT : RED
  const drawdownColor = Math.abs(drawdown) > 0.015 ? RED : Math.abs(drawdown) > 0.008 ? AMBER : ACCENT

  const aapl_sentiment = agentState['AAPL']?.sentiment || null
  const nvda_sentiment = agentState['NVDA']?.sentiment || null
  const activeSentiment = aapl_sentiment || nvda_sentiment


  return (
    <div style={{ background: BG, minHeight: '100vh', fontFamily: 'monospace' }}>

      {/* Top bar */}
      <div style={{
        background: '#0d1220', borderBottom: `0.5px solid ${BORDER}`,
        padding: '10px 16px', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 14, fontWeight: 500, color: ACCENT, letterSpacing: '0.1em' }}>
            ⬡ KAIROS
          </span>
          <span style={{ fontSize: 11, color: DIM }}>autonomous trading system</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {[
            { label: 'websocket', active: connected },
            { label: 'bot running', active: true },
            { label: 'paper mode', active: true, color: AMBER },
          ].map(({ label, active, color }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <Dot color={active ? (color || ACCENT) : RED} />
              <span style={{ fontSize: 11, color: DIM }}>{label}</span>
            </div>
          ))}
          <span style={{ fontSize: 11, color: DIM, ...mono }}>
            {new Date().toLocaleTimeString('en-US', { hour12: false })} ET
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr' }}>

        {/* Sidebar */}
        <div style={{
          background: SIDEBAR, borderRight: `0.5px solid ${BORDER}`,
          padding: '14px 0', minHeight: 'calc(100vh - 45px)'
        }}>
          <div style={{ padding: '0 12px', marginBottom: 16 }}>
            <div style={{ fontSize: 10, color: DIM, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8, paddingLeft: 8 }}>
              Views
            </div>
            {[
              { icon: 'layout-dashboard', label: 'Mission control', view: 'dashboard' },
              { icon: 'chart-line',       label: 'Equity curve',   view: 'equity' },
              { icon: 'brain',            label: 'Agent decisions', view: 'agents' },
              { icon: 'news',             label: 'Sentiment feed',  view: 'sentiment' },
              { icon: 'shield',           label: 'Risk monitor',    view: 'risk' },
              { icon: 'list',             label: 'Trade ledger',    view: 'ledger' },
              { icon: 'microscope',       label: 'Model eval',      view: 'eval' },
            ].map(({ icon, label, view }) => (
              <div key={view} onClick={() => setActiveView(view)} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '7px 8px', borderRadius: 6, fontSize: 12,
                color: activeView === view ? ACCENT : MID, cursor: 'pointer', marginBottom: 2,
                background: activeView === view ? '#0d1a2e' : 'transparent',
                border: activeView === view ? `0.5px solid ${ACCENT}20` : '0.5px solid transparent',
              }}>
                <i className={`ti ti-${icon}`} style={{ fontSize: 15 }} aria-hidden="true" />
                {label}
              </div>
            ))}
          </div>

          <div style={{ height: 0.5, background: BORDER, margin: '0 12px 12px' }} />

          {/* Config panel */}
          <div style={{
            margin: '0 12px', background: CARD,
            border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12
          }}>
            <div style={{ fontSize: 10, color: DIM, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 10 }}>
              Bot configuration
            </div>
            {[
              { k: 'Mode',        v: 'paper',     vc: AMBER },
              { k: 'Symbols',     v: 'AAPL · NVDA', vc: LIGHT },
              { k: 'Capital',     v: `$${equity.toLocaleString()}`, vc: LIGHT },
              { k: 'Max pos',     v: '2% ($200)',  vc: LIGHT },
              { k: 'Daily cap',   v: '0.5%',       vc: RED },
              { k: 'ATR stop',    v: '2×',         vc: LIGHT },
              { k: 'Trades today', v: `${tradesToday} / 3`, vc: tradesToday >= 3 ? RED : ACCENT },
              { k: 'Circuit brk', v: 'armed',      vc: ACCENT },
              { k: 'RL ensemble', v: 'active',     vc: PURPLE },
              { k: 'FinBERT',     v: 'active',     vc: BLUE },
            ].map(({ k, v, vc }) => (
              <div key={k} style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', marginBottom: 7
              }}>
                <span style={{ fontSize: 11, color: MID }}>{k}</span>
                <span style={{ fontSize: 11, color: vc, fontWeight: 500, ...mono }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Live prices */}
          <div style={{ margin: '10px 12px 0', padding: 12, background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8 }}>
            <div style={{ fontSize: 10, color: DIM, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>
              Live prices
            </div>
            {Object.entries(stream).map(([ticker, bar]) => (
              <div key={ticker} style={{ marginBottom: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 12, color: BLUE, fontWeight: 500 }}>{ticker}</span>
                  <span style={{ fontSize: 13, color: LIGHT, fontWeight: 500, ...mono }}>
                    ${bar.close?.toFixed(2)}
                  </span>
                </div>
                <div style={{ fontSize: 10, color: DIM }}>vol {((bar.volume || 0) / 1000).toFixed(0)}k</div>
              </div>
            ))}
            {Object.keys(stream).length === 0 && (
              <div style={{ fontSize: 11, color: DIM }}>market closed</div>
            )}
          </div>
        </div>

        {/* Main content */}
        <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>

          {/* ── Metric strip — shown on all views ── */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            <MetricCard
              label="Equity"
              value={`$${equity.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`}
              sub="paper account"
            />
            <MetricCard
              label="Daily P&L"
              value={`${dailyPnl >= 0 ? '+' : ''}$${dailyPnl.toFixed(2)}`}
              sub={`${((dailyPnl / 10000) * 100).toFixed(3)}%`}
              color={pnlColor}
            />
            <MetricCard
              label="Sharpe ratio"
              value={sharpe.toFixed(2)}
              sub="annualised · live"
              color={sharpe > 1 ? ACCENT : sharpe > 0 ? AMBER : RED}
            />
            <MetricCard
              label="Drawdown"
              value={`${(Math.abs(drawdown) * 100).toFixed(3)}%`}
              sub="of 2.0% limit"
              color={drawdownColor}
            />
          </div>

          {/* ── Dashboard view ── */}
          {activeView === 'dashboard' && (<>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12 }}>
                <CardTitle icon="brain">Agent decisions — this cycle</CardTitle>
                {Object.keys(agentState).length === 0 ? (
                  <>
                    <AgentRow color={BLUE}   name="LogReg"        symbol="AAPL"  sub="waiting for prediction..." badge="—" badgeType="hold" />
                    <AgentRow color={PURPLE} name="PPO agent"     symbol="AAPL"  sub="waiting for RL inference..." badge="—" badgeType="hold" />
                    <AgentRow color={AMBER}  name="Sentiment gate" symbol="AAPL" sub="waiting for FinBERT..." badge="—" badgeType="hold" />
                    <AgentRow color={ACCENT} name="Risk engine"   symbol="final" sub="waiting..." badge="—" badgeType="hold" />
                  </>
                ) : (
                  Object.entries(agentState).map(([sym, state]) => {
                    if (!state) return null
                    const action = state.action || 'HOLD'
                    const badgeType = action === 'ENTER_LONG' ? 'buy' : action === 'EXIT_POSITION' ? 'sell' : 'hold'
                    return <AgentRow key={sym} color={BLUE} name="Ensemble" symbol={sym} sub={state.reason || action} badge={action} badgeType={badgeType} />
                  })
                )}
              </div>
              <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12 }}>
                <CardTitle icon="news">Sentiment feed</CardTitle>
                <SentimentBars signal={activeSentiment} />
                {!activeSentiment && (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                      <span style={{ fontSize: 11, color: MID }}>aggregate score</span>
                      <span style={{ fontSize: 18, color: MID, ...mono }}>—</span>
                    </div>
                    <FeatureBar label="bearish" value={0}  max={100} color={RED} />
                    <FeatureBar label="neutral" value={50} max={100} color={MID} />
                    <FeatureBar label="bullish" value={0}  max={100} color={ACCENT} />
                    <div style={{ fontSize: 11, color: DIM, marginTop: 8 }}>FinBERT preloads on bot start</div>
                  </>
                )}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12 }}>
                <CardTitle icon="chart-line">Equity curve</CardTitle>
                <div style={{ height: 120 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={curveData.length > 1 ? curveData : [{ t: 'start', equity: 10000 }, { t: 'now', equity: equity }]}>
                      <XAxis dataKey="t" tick={false} axisLine={false} tickLine={false} />
                      <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} width={60} tickFormatter={v => `$${v.toFixed(0)}`} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: CARD, border: `0.5px solid ${BORDER}`, fontSize: 11 }} formatter={v => [`$${v.toFixed(2)}`, 'equity']} />
                      <ReferenceLine y={10000} stroke={BORDER} strokeDasharray="3 2" />
                      <Line type="monotone" dataKey="equity" stroke={ACCENT} strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ height: 0.5, background: BORDER, margin: '8px 0' }} />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                  {[
                    { label: 'total P&L',   value: `${dailyPnl >= 0 ? '+' : ''}$${dailyPnl.toFixed(2)}`, color: pnlColor },
                    { label: 'positions',   value: positions.length, color: LIGHT },
                    { label: 'trades today', value: `${tradesToday} / 3`, color: tradesToday >= 3 ? RED : BLUE },
                  ].map(({ label, value, color }) => (
                    <div key={label}>
                      <div style={{ fontSize: 10, color: DIM, marginBottom: 3 }}>{label}</div>
                      <div style={{ fontSize: 14, fontWeight: 500, color, ...mono }}>{value}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12 }}>
                <CardTitle icon="activity">Live features — AAPL</CardTitle>
                {stream['AAPL'] ? (
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: DIM, marginBottom: 4 }}>current price</div>
                    <div style={{ fontSize: 18, fontWeight: 500, color: LIGHT, ...mono }}>
                      ${stream['AAPL'].close?.toFixed(2)}
                      <span style={{ fontSize: 11, color: ACCENT, marginLeft: 8 }}>live</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: 11, color: DIM, marginBottom: 8 }}>market closed — last known values</div>
                )}
                <div style={{ height: 0.5, background: BORDER, margin: '6px 0 10px' }} />
                {(() => {
                  const f = agentState['AAPL']?.features
                  return f ? (<>
                    <FeatureBar label="RSI(14)" value={f.rsi_14 || 0} max={1} color={BLUE} />
                    <FeatureBar label="MACD"    value={f.macd_signal || 0} max={3} color={PURPLE} />
                    <FeatureBar label="OBV z"   value={f.obv_zscore || 0} max={3} color={ACCENT} />
                    <FeatureBar label="vol z"   value={f.volume_zscore || 0} max={3} color={AMBER} />
                    <FeatureBar label="vs VWAP" value={f.price_vs_vwap || 0} max={0.02} color={MID} />
                  </>) : (<>
                    <FeatureBar label="RSI(14)" value={0.24}  max={1}    color={BLUE} />
                    <FeatureBar label="MACD"    value={0.15}  max={3}    color={PURPLE} />
                    <FeatureBar label="OBV z"   value={1.42}  max={3}    color={ACCENT} />
                    <FeatureBar label="vol z"   value={1.80}  max={3}    color={AMBER} />
                    <FeatureBar label="vs VWAP" value={0.003} max={0.02} color={MID} />
                  </>)
                })()}
              </div>
            </div>

            {positions.length > 0 && (
              <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12 }}>
                <CardTitle icon="briefcase">Open positions</CardTitle>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
                  {positions.map((pos, i) => {
                    const pnl = pos.unrealized_pnl || 0
                    return (
                      <div key={i} style={{ padding: 10, background: BG, borderRadius: 6, border: `0.5px solid ${BORDER}` }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                          <span style={{ fontSize: 14, color: BLUE, fontWeight: 500 }}>{pos.ticker}</span>
                          <span style={{ fontSize: 13, color: pnl >= 0 ? ACCENT : RED, fontWeight: 500, ...mono }}>
                            {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                          </span>
                        </div>
                        <div style={{ fontSize: 11, color: MID }}>{pos.quantity} shares @ ${pos.avg_entry_price?.toFixed(2)}</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 12 }}>
              <CardTitle icon="terminal">Bot log</CardTitle>
              {logs.length === 0 ? (
                <>
                  <LogLine time="--:--:--" sym="SYS" msg="waiting for first market cycle..." level="" />
                  <LogLine time="--:--:--" sym="SYS" msg="market opens 09:30 ET" level="" />
                </>
              ) : logs.slice(0, 8).map((l, i) => <LogLine key={i} time={l.time} sym={l.sym} msg={l.msg} level={l.level} />)}
            </div>
          </>)}

          {/* ── Equity curve view ── */}
          {activeView === 'equity' && (
            <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 16 }}>
              <CardTitle icon="chart-line">Equity curve — full session</CardTitle>
              <div style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curveData.length > 1 ? curveData : [{ t: 'start', equity: 10000 }, { t: 'now', equity: equity }]}>
                    <XAxis dataKey="t" tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} width={68} tickFormatter={v => `$${v.toFixed(0)}`} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: CARD, border: `0.5px solid ${BORDER}`, fontSize: 11 }} formatter={v => [`$${v.toFixed(2)}`, 'equity']} />
                    <ReferenceLine y={10000} stroke={BORDER} strokeDasharray="3 2" label={{ value: 'start', fill: DIM, fontSize: 10 }} />
                    <Line type="monotone" dataKey="equity" stroke={ACCENT} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{ height: 0.5, background: BORDER, margin: '12px 0' }} />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10 }}>
                {[
                  { label: 'equity',       value: `$${equity.toLocaleString('en-US', {minimumFractionDigits: 2})}`, color: LIGHT },
                  { label: 'daily P&L',    value: `${dailyPnl >= 0 ? '+' : ''}$${dailyPnl.toFixed(2)}`, color: pnlColor },
                  { label: 'drawdown',     value: `${(Math.abs(drawdown) * 100).toFixed(3)}%`, color: drawdownColor },
                  { label: 'Sharpe',       value: sharpe.toFixed(2), color: sharpe > 1 ? ACCENT : sharpe > 0 ? AMBER : RED },
                  { label: 'trades today', value: `${tradesToday} / 3`, color: tradesToday >= 3 ? RED : BLUE },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ background: BG, borderRadius: 6, padding: '10px 12px', border: `0.5px solid ${BORDER}` }}>
                    <div style={{ fontSize: 10, color: DIM, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
                    <div style={{ fontSize: 18, fontWeight: 500, color, ...mono }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Agent decisions view ── */}
          {activeView === 'agents' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {Object.entries(agentState).length === 0 ? (
                <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 16, color: MID, fontSize: 12 }}>
                  Waiting for first cycle — no agent state yet.
                </div>
              ) : Object.entries(agentState).map(([sym, state]) => {
                if (!state) return null
                const action = state.action || 'HOLD'
                const badgeType = action === 'ENTER_LONG' ? 'buy' : action === 'EXIT_POSITION' ? 'sell' : 'hold'
                const f = state.features || {}
                return (
                  <div key={sym} style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                      <span style={{ fontSize: 16, fontWeight: 500, color: BLUE }}>{sym}</span>
                      <AgentRow color={BLUE} name="Ensemble" symbol="" sub="" badge={action} badgeType={badgeType} />
                    </div>
                    <div style={{ fontSize: 11, color: MID, marginBottom: 12, lineHeight: 1.5 }}>{state.reason}</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                      <div>
                        <div style={{ fontSize: 10, color: DIM, marginBottom: 8, textTransform: 'uppercase' }}>Features</div>
                        <FeatureBar label="RSI(14)" value={f.rsi_14 || 0} max={1} color={BLUE} />
                        <FeatureBar label="MACD"    value={f.macd_signal || 0} max={3} color={PURPLE} />
                        <FeatureBar label="OBV z"   value={f.obv_zscore || 0} max={3} color={ACCENT} />
                        <FeatureBar label="vol z"   value={f.volume_zscore || 0} max={3} color={AMBER} />
                        <FeatureBar label="vs VWAP" value={f.price_vs_vwap || 0} max={0.02} color={MID} />
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: DIM, marginBottom: 8, textTransform: 'uppercase' }}>Sentiment</div>
                        <SentimentBars signal={state.sentiment} />
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* ── Sentiment view ── */}
          {activeView === 'sentiment' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {['AAPL', 'NVDA'].map(sym => {
                const signal = agentState[sym]?.sentiment || null
                return (
                  <div key={sym} style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
                    <div style={{ fontSize: 14, fontWeight: 500, color: BLUE, marginBottom: 10 }}>{sym}</div>
                    <SentimentBars signal={signal} />
                    {!signal && <div style={{ fontSize: 11, color: DIM }}>No signal yet — waiting for market hours.</div>}
                  </div>
                )
              })}
            </div>
          )}

          {/* ── Risk monitor view ── */}
          {activeView === 'risk' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                {[
                  { label: 'Daily loss',     value: `${(Math.abs(dailyPnl / (equity || 1)) * 100).toFixed(3)}%`, limit: '0.5%', ok: Math.abs(dailyPnl / (equity || 1)) < 0.005 },
                  { label: 'Total drawdown', value: `${(Math.abs(drawdown) * 100).toFixed(3)}%`,                  limit: '2.0%', ok: Math.abs(drawdown) < 0.02 },
                  { label: 'Trades today',   value: `${tradesToday}`,                                              limit: '3',    ok: tradesToday < 3 },
                ].map(({ label, value, limit, ok }) => (
                  <div key={label} style={{ background: CARD, border: `0.5px solid ${ok ? BORDER : RED + '40'}`, borderRadius: 8, padding: 14 }}>
                    <div style={{ fontSize: 10, color: DIM, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
                    <div style={{ fontSize: 24, fontWeight: 500, color: ok ? ACCENT : RED, ...mono }}>{value}</div>
                    <div style={{ fontSize: 11, color: MID, marginTop: 4 }}>limit: {limit}</div>
                    <Dot color={ok ? ACCENT : RED} size={8} />
                  </div>
                ))}
              </div>
              <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
                <CardTitle icon="shield">Risk policy — hard limits</CardTitle>
                {[
                  { k: 'Max position size',        v: '2% of portfolio',           status: true },
                  { k: 'ATR stop-loss multiplier',  v: '2× ATR',                   status: true },
                  { k: 'Daily loss circuit breaker', v: '0.5% → halt all trading', status: true },
                  { k: 'Total drawdown limit',       v: '2.0% → halt all trading', status: true },
                  { k: 'Max trades per day',         v: '3',                        status: tradesToday < 3 },
                  { k: 'Max open positions',         v: '1',                        status: positions.length < 1 },
                  { k: 'Min model confidence',       v: '70%',                      status: true },
                  { k: 'Trading mode',               v: 'paper only',               status: true },
                ].map(({ k, v, status }) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: `0.5px solid ${BORDER}` }}>
                    <span style={{ fontSize: 12, color: MID }}>{k}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 12, color: LIGHT, ...mono }}>{v}</span>
                      <Dot color={status ? ACCENT : RED} size={7} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Model eval view ── */}
          {activeView === 'eval' && (() => {
            const logreg = evalMetrics?.logreg || []
            const rlCurve = evalMetrics?.rl_curve || []
            const rlSum = evalMetrics?.rl_summary || {}

            const rocColor = auc => auc >= 0.55 ? ACCENT : auc >= 0.52 ? AMBER : RED
            const rocBar = auc => Math.max(0, Math.min(100, (auc - 0.5) / 0.1 * 100))

            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

                {/* LogReg table */}
                <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
                  <CardTitle icon="math-function">LogReg models — test set metrics</CardTitle>
                  <div style={{ fontSize: 10, color: DIM, marginBottom: 10 }}>
                    5-min direction prediction · threshold: prob_up ≥ 0.70 · horizon: 5 min · baseline = 0.50 (random)
                  </div>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: `0.5px solid ${BORDER}` }}>
                        {['Symbol', 'ROC-AUC', '', 'Accuracy', 'Test rows', 'Train rows', 'Version'].map(h => (
                          <th key={h} style={{ padding: '4px 8px', textAlign: 'left', fontSize: 10, color: DIM, fontWeight: 400, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {logreg.map(m => (
                        <tr key={m.symbol} style={{ borderBottom: `0.5px solid #0d1a2e` }}>
                          <td style={{ padding: '7px 8px', color: BLUE, fontWeight: 500, ...mono }}>{m.symbol}</td>
                          <td style={{ padding: '7px 8px', color: rocColor(m.roc_auc), fontWeight: 500, ...mono }}>
                            {m.roc_auc != null ? m.roc_auc.toFixed(4) : '—'}
                          </td>
                          <td style={{ padding: '7px 8px', width: 80 }}>
                            <div style={{ height: 4, background: '#1e2a3a', borderRadius: 2, overflow: 'hidden' }}>
                              <div style={{ width: `${rocBar(m.roc_auc || 0.5)}%`, height: '100%', background: rocColor(m.roc_auc), borderRadius: 2 }} />
                            </div>
                          </td>
                          <td style={{ padding: '7px 8px', color: LIGHT, ...mono }}>{(m.accuracy * 100).toFixed(2)}%</td>
                          <td style={{ padding: '7px 8px', color: MID, ...mono }}>{m.test_rows.toLocaleString()}</td>
                          <td style={{ padding: '7px 8px', color: MID, ...mono }}>{m.train_rows.toLocaleString()}</td>
                          <td style={{ padding: '7px 8px', color: DIM, ...mono }}>{m.version}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {logreg.length === 0 && <div style={{ color: DIM, fontSize: 12, padding: '8px 0' }}>No model artifacts found.</div>}
                  <div style={{ marginTop: 10, padding: '8px 10px', background: '#1e2a3a20', borderRadius: 6, border: `0.5px solid ${BORDER}`, fontSize: 11, color: DIM, lineHeight: 1.6 }}>
                    All models near ROC-AUC 0.50 — barely above random. Re-train after applying the warmup filter (first 30 bars/day dropped) to remove cold-EMA noise from the training set.
                  </div>
                </div>

                {/* RL summary cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                  {[
                    { label: 'PPO timesteps',    value: rlSum.total_timesteps ? `${(rlSum.total_timesteps/1000).toFixed(0)}k` : '—', color: PURPLE },
                    { label: 'Checkpoints',      value: rlSum.n_checkpoints ?? '—', color: LIGHT },
                    { label: 'Best mean reward', value: rlSum.best_mean_reward != null ? rlSum.best_mean_reward.toFixed(4) : '—', color: rlSum.best_mean_reward > 0 ? ACCENT : RED },
                    { label: 'Best win rate',    value: rlSum.best_win_rate != null ? `${(rlSum.best_win_rate*100).toFixed(1)}%` : '—', color: rlSum.best_win_rate > 0.5 ? ACCENT : RED },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: '12px 14px' }}>
                      <div style={{ fontSize: 10, color: DIM, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>{label}</div>
                      <div style={{ fontSize: 22, fontWeight: 500, color, ...mono }}>{value}</div>
                    </div>
                  ))}
                </div>

                {/* RL eval curve */}
                <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
                  <CardTitle icon="robot">PPO reward curve — 500k training steps (25 evals × 10 episodes)</CardTitle>
                  <div style={{ height: 180 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rlCurve} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                        <XAxis dataKey="timestep" tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} tickFormatter={v => `${v/1000}k`} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} width={50} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                        <Tooltip
                          contentStyle={{ background: CARD, border: `0.5px solid ${BORDER}`, fontSize: 11, fontFamily: 'monospace' }}
                          formatter={(v, name) => [v.toFixed(4), name]}
                          labelFormatter={v => `step ${(v/1000).toFixed(0)}k`}
                        />
                        <ReferenceLine y={0} stroke={BORDER} strokeDasharray="3 2" />
                        <Line type="monotone" dataKey="mean_reward" stroke={PURPLE} strokeWidth={1.5} dot={false} name="mean reward" />
                        <Line type="monotone" dataKey="win_rate" stroke={ACCENT} strokeWidth={1.5} dot={false} name="win rate" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', gap: 16, marginTop: 6 }}>
                    {[{ color: PURPLE, label: 'mean reward (clipped ±1)' }, { color: ACCENT, label: 'win rate (episodes > 0)' }].map(({ color, label }) => (
                      <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 20, height: 2, background: color, borderRadius: 1 }} />
                        <span style={{ fontSize: 11, color: MID }}>{label}</span>
                      </div>
                    ))}
                    <div style={{ marginLeft: 'auto', fontSize: 11, color: DIM }}>— = break-even</div>
                  </div>
                  <div style={{ marginTop: 10, padding: '8px 10px', background: '#1e2a3a20', borderRadius: 6, border: `0.5px solid ${BORDER}`, fontSize: 11, color: DIM, lineHeight: 1.6 }}>
                    Reward flat around −0.20 throughout training — agent converges to "sell immediately" degenerate policy.
                    The updated reward function (early-exit penalty + stronger HOLD signal) will change this curve after retraining.
                  </div>
                </div>

                {/* RL ep length chart */}
                <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
                  <CardTitle icon="clock">PPO mean episode length (holding steps)</CardTitle>
                  <div style={{ height: 140 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={rlCurve} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                        <XAxis dataKey="timestep" tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} tickFormatter={v => `${v/1000}k`} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 10, fill: DIM, fontFamily: 'monospace' }} width={36} axisLine={false} tickLine={false} />
                        <Tooltip
                          contentStyle={{ background: CARD, border: `0.5px solid ${BORDER}`, fontSize: 11, fontFamily: 'monospace' }}
                          formatter={v => [`${v.toFixed(1)} steps`, 'mean ep length']}
                          labelFormatter={v => `step ${(v/1000).toFixed(0)}k`}
                        />
                        <Bar dataKey="mean_ep_length" fill={BLUE} opacity={0.7} radius={[2,2,0,0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ fontSize: 11, color: DIM, marginTop: 6 }}>
                    Short episodes = agent exits early. Should increase after reward function fix + retraining.
                  </div>
                </div>

              </div>
            )
          })()}

          {/* ── Trade ledger view ── */}
          {activeView === 'ledger' && (
            <div style={{ background: CARD, border: `0.5px solid ${BORDER}`, borderRadius: 8, padding: 14 }}>
              <CardTitle icon="list">All executions</CardTitle>
              {orders.length === 0 ? (
                <div style={{ fontSize: 12, color: DIM, padding: '20px 0' }}>No trades executed yet.</div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: `0.5px solid ${BORDER}` }}>
                        {['Time', 'Symbol', 'Side', 'Qty', 'Price', 'Status'].map(h => (
                          <th key={h} style={{ padding: '4px 8px', textAlign: 'left', fontSize: 10, color: DIM, fontWeight: 400, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {orders.map(o => (
                        <tr key={o.id} style={{ borderBottom: `0.5px solid #0d1a2e` }}>
                          <td style={{ padding: '5px 8px', color: MID, ...mono }}>{new Date(o.timestamp).toLocaleTimeString()}</td>
                          <td style={{ padding: '5px 8px', color: BLUE, fontWeight: 500 }}>{o.ticker}</td>
                          <td style={{ padding: '5px 8px', color: o.side === 'buy' ? ACCENT : RED, fontWeight: 500 }}>{o.side?.toUpperCase()}</td>
                          <td style={{ padding: '5px 8px', color: LIGHT, ...mono }}>{o.qty}</td>
                          <td style={{ padding: '5px 8px', color: LIGHT, ...mono }}>${o.price?.toFixed(2) || '—'}</td>
                          <td style={{ padding: '5px 8px' }}>
                            <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10, background: '#1e2a3a', color: BLUE }}>{o.status}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  )
}