import { useEffect, useState } from 'react'

function App() {
  const [data, setData] = useState({ portfolio: null, recent_orders: [] })
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/dashboard/summary')
        if (!response.ok) throw new Error("Network response was not ok")
        const result = await response.json()
        setData(result)
        setError(null)
      } catch (error) {
        console.error("Fetch error:", error)
        setError("Unable to connect to the trading backend.")
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000) // Poll every 5 seconds
    return () => clearInterval(interval)
  }, [])

  if (error) return <div className="p-10 text-red-500 font-bold">{error}</div>
  if (!data.portfolio) return <div className="p-10 text-gray-400">Booting TradeOps AI...</div>

  const pnlColor = data.portfolio.daily_pnl >= 0 ? "text-green-400" : "text-red-400"

  return (
    <div className="min-h-screen p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-extrabold text-blue-400">TradeOps AI Mission Control</h1>
        <p className="text-gray-400">Autonomous Intraday Agent</p>
      </header>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
        <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
          <h2 className="text-gray-400 text-sm font-semibold mb-1">Live Equity</h2>
          <p className="text-3xl font-bold">${data.portfolio.equity.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
          <h2 className="text-gray-400 text-sm font-semibold mb-1">Available Cash</h2>
          <p className="text-3xl font-bold">${data.portfolio.cash.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
          <h2 className="text-gray-400 text-sm font-semibold mb-1">Daily PnL</h2>
          <p className={`text-3xl font-bold ${pnlColor}`}>
            {data.portfolio.daily_pnl >= 0 ? "+" : ""}${data.portfolio.daily_pnl.toFixed(2)}
          </p>
        </div>
        <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
          <h2 className="text-gray-400 text-sm font-semibold mb-1">Total Drawdown</h2>
          <p className="text-3xl font-bold text-red-400">
            {(data.portfolio.drawdown_pct * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Orders Table */}
      <div className="bg-slate-800 rounded-lg shadow-lg border border-slate-700 overflow-hidden">
        <div className="p-6 border-b border-slate-700">
          <h2 className="text-xl font-bold">Recent Executions</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-900 text-gray-400 text-sm">
                <th className="p-4 font-semibold">Time</th>
                <th className="p-4 font-semibold">Symbol</th>
                <th className="p-4 font-semibold">Side</th>
                <th className="p-4 font-semibold">Qty</th>
                <th className="p-4 font-semibold">Price</th>
                <th className="p-4 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_orders.length === 0 ? (
                <tr>
                  <td colSpan="6" className="p-4 text-center text-gray-500">No trades executed yet.</td>
                </tr>
              ) : (
                data.recent_orders.map((order) => (
                  <tr key={order.id} className="border-b border-slate-700 hover:bg-slate-700/50">
                    <td className="p-4 text-sm text-gray-300">{new Date(order.timestamp).toLocaleTimeString()}</td>
                    <td className="p-4 font-bold">{order.ticker}</td>
                    <td className={`p-4 font-bold ${order.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                      {order.side.toUpperCase()}
                    </td>
                    <td className="p-4">{order.qty}</td>
                    <td className="p-4">${order.price?.toFixed(2) || '---'}</td>
                    <td className="p-4">
                      <span className="bg-blue-900/50 text-blue-300 py-1 px-3 rounded-full text-xs font-semibold">
                        {order.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default App