# ⬡ Kairos — Autonomous Intraday Trading System

> An autonomous trading system that combines reinforcement learning (PPO) and logistic regression to make real-time buy/sell decisions on US equities without human intervention.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-RL-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What Kairos Does

Kairos runs a continuous autonomous loop during US market hours:

```
Alpaca WebSocket → Redis → Feature Engine → RL + LogReg Ensemble
    → Risk Gate → Alpaca Paper API → Postgres → Live Dashboard
```

No human approves trades. The system observes, decides, gatekeeps itself through hard-coded risk rules, and executes — every 60 seconds, all day.

---

## Architecture

Kairos is a multi-agent system built in five layers. Each layer only talks to the one below it. No shortcuts.

```
┌─────────────────────────────────────────────┐
│  Layer 1 — Data                             │
│  Alpaca WS → Redis streams → Postgres       │
├─────────────────────────────────────────────┤
│  Layer 2 — Signal (multi-agent ensemble)    │
│  PredictionAgent (LogReg)                   │
│  RLAgent (PPO)          ┐ both must agree   │
│  DecisionAgent ─────────┘ to trade          │
│  PortfolioAgent (live state sync)           │
├─────────────────────────────────────────────┤
│  Layer 3 — Risk (hard-coded, never learned) │
│  ATR stop-loss · drawdown circuit breaker   │
│  Position sizing · daily trade limit        │
├─────────────────────────────────────────────┤
│  Layer 4 — Execution                        │
│  ExecutionAgent · PaperBroker · trade ledger│
├─────────────────────────────────────────────┤
│  Layer 5 — Dashboard + Alerts               │
│  FastAPI WebSocket → React live UI          │
│  Slack + email on circuit breaker / fills   │
└─────────────────────────────────────────────┘
```

---

## The 5 Agents

| Agent | File | Responsibility |
|---|---|---|
| `PredictionAgent` | `app/agents/prediction_agent.py` | Runs LogReg model → `probability_up` score |
| `RLAgent` | `app/rl/rl_agent.py` | Runs PPO policy → HOLD/SELL signal |
| `DecisionAgent` | `app/agents/decision_agent.py` | Ensemble coordinator — both models must agree |
| `PortfolioAgent` | `app/agents/portfolio_agent.py` | Syncs live equity, positions, drawdown from Alpaca |
| `ExecutionAgent` | `app/agents/execution_agent.py` | Sizes orders, submits via PaperBroker, logs fills |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (async), Python 3.11+ |
| RL Agent | Stable-Baselines3 (PPO), PyTorch, Gymnasium |
| ML Baseline | scikit-learn (Logistic Regression pipeline) |
| Data stream | Alpaca WebSocket API, alpaca-py |
| Feature store | Redis (live state), PostgreSQL (history) |
| ORM / DB | SQLAlchemy async, asyncpg, Alembic |
| Infrastructure | Docker Compose (Postgres 16 + Redis 7) |
| Dashboard | React 19, Recharts, FastAPI WebSocket |
| Alerts | Slack incoming webhooks, Gmail SMTP |
| Build | Vite, pip, pyproject.toml |

---

## Project Structure

```
Kairos/
├── backend/
│   ├── app/
│   │   ├── agents/           # 5 autonomous agents
│   │   ├── api/              # FastAPI routers + WebSocket endpoint
│   │   ├── core/             # Config, Redis client, alerting
│   │   ├── data/             # Alpaca bar service + live WS stream
│   │   ├── db/               # SQLAlchemy models + async session
│   │   ├── execution/        # PaperBroker + AutonomousBot loop
│   │   ├── features/         # Feature engineering
│   │   ├── models/           # Dataset builder + LogReg model
│   │   ├── risk/             # RiskEngine + RiskPolicy
│   │   └── rl/               # TradingEnv (Gymnasium) + PPO wrapper
│   ├── scripts/              # Training, data loading, validation
│   ├── artifacts/
│   │   ├── models/           # LogReg .joblib artifacts
│   │   └── rl/               # PPO .zip checkpoints (11 saved)
│   └── pyproject.toml
├── frontend/
│   └── src/App.jsx           # Live WebSocket dashboard
└── infra/
    └── docker-compose.yml
```

---

## How It Works

### 1. Live Data Pipeline
`stream_client.py` opens a persistent WebSocket to Alpaca IEX. Every 1-minute bar writes to Redis (sub-millisecond reads for the agent) and Postgres (durable history for training).

### 2. Feature Engineering
5 features computed per bar from a rolling window:

| Feature | What it measures |
|---|---|
| `return_1m` | 1-minute price momentum |
| `return_5m` | 5-minute price momentum |
| `volatility_10m` | Rolling 10-min standard deviation |
| `volume_zscore` | Volume vs recent average |
| `price_vs_vwap` | Price position relative to VWAP |

### 3. Multi-Agent Ensemble
Two independent models vote on every signal. Both must agree to trigger a trade — disagreement produces HOLD.

**LogReg** (`PredictionAgent`): supervised classifier predicting 5-min price direction. Entry threshold: `probability_up ≥ 0.70`, `confidence ≥ 0.70`.

**PPO** (`RLAgent`): reinforcement learning agent trained in a Gymnasium environment. Observation: 5 market features + holding time + unrealized P&L. Reward: clipped percentage return × 100.

**Ensemble rule:**
```
LogReg ENTER + RL bullish  → ENTER_LONG  (both agree)
LogReg EXIT  + RL bearish  → EXIT        (both agree)
Any disagreement           → HOLD        (uncertainty)
```

### 4. Risk Gate (hard-coded, never learned by RL)

| Rule | Value | Trigger |
|---|---|---|
| Max position size | 2% of portfolio | Per trade |
| ATR stop-loss | 2× ATR below entry | Continuous monitoring |
| Daily loss circuit breaker | 0.5% | Halts all trading |
| Total drawdown limit | 2% | Halts all trading |
| Max trades per day | 3 | Per cycle |
| Max open positions | 1 | Per entry |
| Min model confidence | 70% | Per signal |
| Trading mode | Paper only | Always |

### 5. Execution
Market orders via Alpaca paper API. Every fill logged to `paper_orders`, `paper_fills`, `positions`, `portfolio_snapshots`.

### 6. Dashboard + Alerts
FastAPI pushes to React via WebSocket every second. Shows: live equity curve, Sharpe ratio, positions, agent signals, risk gauge, order blotter.

Alerts fire on: circuit breaker trigger, ATR stop-loss execution, trade fill. Delivered via Slack and/or email.

---

## Data

| Symbol | Bars | Features | Date Range |
|---|---|---|---|
| AAPL | 52,556 | 52,166 | Dec 2025 → Jun 2026 |
| NVDA | 53,605 | 53,209 | Dec 2025 → Jun 2026 |

~130 trading days × 390 1-min bars/day per symbol.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- Alpaca paper trading account (free at alpaca.markets)

### 1. Clone and configure
```bash
git clone https://github.com/UdayKiranJadi/Kairos.git
cd Kairos
cp backend/.env.example backend/.env
# Edit backend/.env — add Alpaca keys and optional alert credentials
```

### 2. Start infrastructure
```bash
cd infra && docker compose up -d
```

### 3. Set up Python environment
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Initialise database and load data
```bash
PYTHONPATH=. python scripts/create_tables.py
PYTHONPATH=. python scripts/seed_symbols.py
PYTHONPATH=. python scripts/load_historical_bars.py \
  --symbols AAPL,NVDA \
  --start 2025-12-01T13:30:00 \
  --end 2026-06-13T20:00:00
PYTHONPATH=. python scripts/build_features.py \
  --symbols AAPL,NVDA \
  --start 2025-12-01T13:30:00 \
  --end 2026-06-13T20:00:00
```

### 5. Train models
```bash
# LogReg baseline
PYTHONPATH=. python scripts/train_prediction_model.py \
  --symbol AAPL --horizon-minutes 5

# PPO RL agent
PYTHONPATH=. python scripts/train_rl_agent.py \
  --symbol AAPL --timesteps 500000 --days-history 200
```

### 6. Run
```bash
# Terminal 1 — backend API
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Terminal 2 — autonomous bot
PYTHONPATH=. python scripts/run_autonomous_bot.py --symbols AAPL,NVDA

# Terminal 3 — dashboard
cd ../frontend && npm install && npm run dev
```

Open `http://localhost:5173`

---

## Environment Variables

```bash
# Alpaca paper trading
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true
ALPACA_DATA_FEED=iex

# Database + cache
DATABASE_URL=postgresql+asyncpg://tradeops:tradeops@localhost:5433/tradeops
REDIS_URL=redis://localhost:6380/0

# Trading
TRADING_MODE=paper

# Alerts (both optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_EMAIL_FROM=you@gmail.com
ALERT_EMAIL_TO=you@gmail.com
ALERT_EMAIL_PASSWORD=xxxx_xxxx_xxxx_xxxx
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/market-data/historical-bars/load` | Load bars from Alpaca |
| GET | `/market-data/historical-bars/{symbol}` | Recent bars |
| POST | `/features/build` | Build features |
| GET | `/features/{symbol}` | Recent features |
| POST | `/predictions/run/latest` | Run LogReg prediction |
| GET | `/predictions/{symbol}` | Recent predictions |
| POST | `/decisions/evaluate/latest` | Risk-checked signal |
| GET | `/dashboard/summary` | Portfolio summary (REST) |
| WS | `/ws/live` | Live dashboard push (1s) |

---

## DB Schema (11 tables)

```
symbols            — tracked tickers
market_bars        — 1-min OHLCV (52k+ rows per symbol)
feature_snapshots  — 5 computed features per bar
predictions        — LogReg model outputs
rl_actions         — RL agent decisions + rewards
risk_checks        — every risk evaluation logged
paper_orders       — submitted orders
paper_fills        — execution fills
positions          — open / closed positions
portfolio_snapshots — equity snapshots (equity curve)
audit_events       — full system event log
```

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| Live data pipeline (WS → Redis → Postgres) | ✅ Working | IEX free feed |
| Feature engineering (5 indicators) | ✅ Working | Per 1-min bar |
| LogReg prediction model | ✅ Working | ROC-AUC ~0.52 |
| PPO RL environment (Gymnasium) | ✅ Built | Forced-entry, 7-dim obs |
| PPO training pipeline | ✅ Working | 500k steps, 11 checkpoints |
| RL + LogReg ensemble | ✅ Wired | Both must agree to trade |
| Risk engine (ATR stop + circuit breaker) | ✅ Working | Hard-coded, not learned |
| Autonomous bot loop | ✅ Running | Paper mode, 60s cycle |
| Live WebSocket dashboard | ✅ Working | Equity curve, Sharpe, signals |
| Slack + email alerts | ✅ Wired | Circuit breaker, stops, fills |
| RL agent eval win rate | ⚠️ 0% (50 episodes) | Reward function needs redesign |
| Live capital deployment | ❌ Not yet | Paper only until Sharpe > 1.0 |

---

## Known Issues & Next Steps

**RL reward collapse** — the PPO agent converges to "sell immediately" on every episode (win rate 0%, mean reward -0.20). Root cause: the forced-entry environment makes holding consistently costly relative to immediate exit, so the agent finds the locally optimal degenerate policy. Fix planned: multi-step reward that explicitly penalises premature exits and rewards holding through profitable moves.

**Planned improvements (in priority order):**
1. RL reward function redesign — multi-step lookahead reward
2. Additional features — RSI(14), MACD, OBV
3. Kelly criterion position sizing
4. Walk-forward backtesting across market regimes
5. VaR portfolio-level risk check
6. TWAP execution for larger orders

---

## Build Log

| Day | What was built |
|---|---|
| Day 1 | Alpaca WebSocket stream → Redis · bot decoupled from REST polling |
| Day 2 | Gymnasium TradingEnv — 7-dim obs, forced-entry design, Sharpe reward |
| Day 3 | PPO training pipeline — 500k steps on 6 months of real AAPL/NVDA data |
| Day 4 | ATR stop-loss + circuit breaker + RL ensemble wired into AutonomousBot |
| Day 5 | FastAPI WebSocket → React dashboard (equity curve, agent signals, risk gauge) |
| Day 6 | Live Sharpe metric on dashboard · Slack/email alert system · 500k PPO retrain |

---

## Honest Assessment

This is a paper trading system built for learning and demonstration. The multi-agent architecture, risk engine, and live data pipeline are production-quality. The RL agent's reward function needs further work before it produces a viable trading policy.

Before any live capital deployment:
- Minimum 3 months paper trading with Sharpe > 1.0
- RL reward function redesign and retraining
- Walk-forward backtesting across multiple market regimes
- Slippage and market impact modelling
- Regulatory compliance review

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Uday Kiran Jadi**
Built day by day as a full-stack autonomous trading system.

GitHub: [UdayKiranJadi/Kairos](https://github.com/UdayKiranJadi/Kairos)