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

Kairos is built in five layers, each only talking to the one below it:

```
┌─────────────────────────────────────────────┐
│  Layer 1 — Data                             │
│  Alpaca WS → Redis streams → Postgres       │
├─────────────────────────────────────────────┤
│  Layer 2 — Signal                           │
│  PPO (RL) + LogReg ensemble                 │
│  Both must agree to trigger a trade         │
├─────────────────────────────────────────────┤
│  Layer 3 — Risk (hard-coded, never learned) │
│  ATR stop-loss · drawdown circuit breaker   │
│  Position sizing · daily trade limit        │
├─────────────────────────────────────────────┤
│  Layer 4 — Execution                        │
│  Alpaca Paper API · fill logging · ledger   │
├─────────────────────────────────────────────┤
│  Layer 5 — Dashboard                        │
│  FastAPI WebSocket → React live UI          │
└─────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (async), Python 3.11+ |
| RL Agent | Stable-Baselines3 (PPO), PyTorch, Gymnasium |
| ML Baseline | scikit-learn (Logistic Regression pipeline) |
| Data stream | Alpaca WebSocket API, alpaca-py |
| Feature store | Redis (live), PostgreSQL (history) |
| ORM / DB | SQLAlchemy async, asyncpg, Alembic |
| Infrastructure | Docker Compose (Postgres 16 + Redis 7) |
| Dashboard | React 19, Recharts, FastAPI WebSocket |
| Build | Vite, uv/pip, pyproject.toml |

---

## Project Structure

```
Kairos/
├── backend/
│   ├── app/
│   │   ├── agents/           # Decision, Prediction, Portfolio, Execution agents
│   │   ├── api/              # FastAPI routers + WebSocket endpoint
│   │   ├── core/             # Config (pydantic-settings), Redis client
│   │   ├── data/             # Alpaca bar service + live WebSocket stream
│   │   ├── db/               # SQLAlchemy models + async session
│   │   ├── execution/        # PaperBroker (Alpaca) + AutonomousBot loop
│   │   ├── features/         # Feature engineering (return, vol, VWAP, OBV)
│   │   ├── models/           # Dataset builder + LogReg prediction model
│   │   ├── risk/             # RiskEngine + RiskPolicy (hard-coded rules)
│   │   └── rl/               # Gymnasium TradingEnv + PPO inference wrapper
│   ├── scripts/              # Training, data loading, validation scripts
│   ├── artifacts/
│   │   ├── models/           # LogReg .joblib artifacts
│   │   └── rl/               # PPO .zip checkpoints + best model
│   └── pyproject.toml
├── frontend/
│   └── src/
│       └── App.jsx           # Live dashboard (WebSocket, equity curve, signals)
└── infra/
    └── docker-compose.yml    # Postgres + Redis
```

---

## How It Works — Step by Step

### 1. Live Data Pipeline
`app/data/stream_client.py` opens a persistent WebSocket to Alpaca's IEX feed. Every time a 1-minute bar closes, it writes to:
- `Redis` — `kairos:bar:AAPL` (latest bar snapshot, sub-millisecond reads)
- `Postgres` — `market_bars` table (durable history for training)

### 2. Feature Engineering
`app/features/feature_builder.py` computes 5 features per bar from a rolling window:
- `return_1m` — 1-minute price return
- `return_5m` — 5-minute price return
- `volatility_10m` — rolling 10-minute standard deviation
- `volume_zscore` — volume relative to recent average
- `price_vs_vwap` — price position relative to VWAP

### 3. Signal Generation — Ensemble
Two independent models vote on each signal:

**LogReg model** (`artifacts/models/`): trained on supervised data — 5 features → predicts probability of price moving up in the next 5 minutes. Entry threshold: `probability_up ≥ 0.70`.

**PPO RL agent** (`artifacts/rl/`): trained in a Gymnasium environment where the agent manages trade exits (forced-entry design). Uses the same 5 features + holding time + unrealized P&L. Reward: clipped percentage return × 100.

**Ensemble rule**: both models must agree (LogReg says enter + RL says hold/bullish) to produce an `ENTER_LONG` signal. Disagreement → `HOLD`. This reduces false positives.

### 4. Risk Gate
Every signal passes through `app/risk/risk_engine.py` before touching the broker. This layer is **hard-coded** — the RL agent cannot learn to bypass it.

| Rule | Value |
|---|---|
| Max position size | 2% of portfolio |
| Daily loss circuit breaker | 0.5% — halts all trading |
| Total drawdown limit | 2% |
| ATR stop-loss multiplier | 2× ATR |
| Max trades per day | 3 |
| Max open positions | 1 |
| Min model confidence | 70% |
| Trading mode | Paper only |

### 5. Execution
`app/execution/paper_broker.py` wraps Alpaca's Trading API. Market orders only (guarantees fill). Every fill is logged to `paper_orders` and `paper_fills` tables in Postgres.

### 6. Live Dashboard
FastAPI pushes portfolio state, agent decisions, equity curve, and live prices to the React frontend via WebSocket at `ws://localhost:8000/ws/live` — updating every second.

---

## Data

| Symbol | Bars | Features | Date Range |
|---|---|---|---|
| AAPL | 52,556 | 52,166 | Dec 2025 → Jun 2026 |
| NVDA | 53,605 | 53,209 | Dec 2025 → Jun 2026 |

~130 trading days × 390 bars/day = ~50,000 rows per symbol. All 1-minute OHLCV from Alpaca IEX feed.

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
# Edit backend/.env with your Alpaca paper keys
```

### 2. Start infrastructure
```bash
cd infra
docker compose up -d
```

### 3. Set up Python environment
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Initialize database and seed data
```bash
PYTHONPATH=. python scripts/create_tables.py
PYTHONPATH=. python scripts/seed_symbols.py

# Load historical bars (requires market to have been open)
PYTHONPATH=. python scripts/load_historical_bars.py \
  --symbols AAPL,NVDA \
  --start 2025-12-01T13:30:00 \
  --end 2026-06-13T20:00:00

# Build features from bars
PYTHONPATH=. python scripts/build_features.py \
  --symbols AAPL,NVDA \
  --start 2025-12-01T13:30:00 \
  --end 2026-06-13T20:00:00
```

### 5. Train models
```bash
# Train LogReg baseline
PYTHONPATH=. python scripts/train_prediction_model.py \
  --symbol AAPL --horizon-minutes 5

# Train RL agent (PPO)
PYTHONPATH=. python scripts/train_rl_agent.py \
  --symbol AAPL --timesteps 500000 --days-history 200
```

### 6. Run the system
```bash
# Terminal 1 — Backend API
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Terminal 2 — Autonomous bot
PYTHONPATH=. python scripts/run_autonomous_bot.py --symbols AAPL,NVDA

# Terminal 3 — Frontend dashboard
cd ../frontend && npm install && npm run dev
```

Open `http://localhost:5173`

---

## Environment Variables

```bash
# backend/.env
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true
ALPACA_DATA_FEED=iex

DATABASE_URL=postgresql+asyncpg://tradeops:tradeops@localhost:5433/tradeops
REDIS_URL=redis://localhost:6380/0

TRADING_MODE=paper
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/market-data/historical-bars/load` | Load bars from Alpaca |
| GET | `/market-data/historical-bars/{symbol}` | List recent bars |
| POST | `/features/build` | Build features from bars |
| GET | `/features/{symbol}` | Get recent features |
| POST | `/predictions/run/latest` | Run LogReg prediction |
| GET | `/predictions/{symbol}` | Recent predictions |
| POST | `/decisions/evaluate/latest` | Evaluate + risk-check signal |
| GET | `/dashboard/summary` | Portfolio summary (REST) |
| WS | `/ws/live` | Live dashboard stream |

---

## DB Schema

```
symbols          — tracked tickers
market_bars      — 1-min OHLCV (52k+ rows per symbol)
feature_snapshots — computed features per bar
predictions      — LogReg model outputs
rl_actions       — RL agent decisions + rewards
risk_checks      — every risk evaluation logged
paper_orders     — submitted orders
paper_fills      — execution fills
positions        — open/closed positions
portfolio_snapshots — equity snapshots for curve
audit_events     — system event log
```

---

## Current Status

| Component | Status |
|---|---|
| Live data pipeline (WebSocket → Redis → Postgres) | ✅ Working |
| Feature engineering (5 indicators) | ✅ Working |
| LogReg prediction model | ✅ Working (ROC-AUC ~0.52) |
| PPO RL environment (Gymnasium) | ✅ Built, validated |
| PPO training pipeline | ✅ Trains on 50k real bars |
| RL + LogReg ensemble | ✅ Wired into bot |
| Risk engine (ATR stop, circuit breaker) | ✅ Working |
| Autonomous bot loop | ✅ Running in paper mode |
| Live WebSocket dashboard | ✅ Working |
| RL agent profitability | 🔄 In progress (needs more data + tuning) |
| Live capital deployment | ❌ Not yet (paper only) |

---

## Build Log

| Day | What was built |
|---|---|
| Day 1 | Alpaca WebSocket stream → Redis → bot decoupled from REST polling |
| Day 2 | Gymnasium TradingEnv — 7-dim obs space, Sharpe reward, forced-entry design |
| Day 3 | PPO training pipeline — 500k steps on 6 months of real AAPL/NVDA data |
| Day 4 | ATR stop-loss + circuit breaker + RL ensemble wired into AutonomousBot |
| Day 5 | FastAPI WebSocket → React live dashboard (equity curve, agent signals, risk gauge) |

---

## Honest Assessment

This is a paper trading system built for learning and demonstration. The LogReg model achieves modest edge (ROC-AUC ~0.52) and the RL agent's performance improves with more data. Before any live capital deployment, the system would need:

- Extended paper trading validation (minimum 3 months, Sharpe > 1.0)
- Slippage and market impact modeling
- Walk-forward backtesting across multiple market regimes
- Position sizing upgrade (Kelly criterion or volatility targeting)
- Regulatory compliance review

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Uday Kiran Jadi**
Built as a full-stack autonomous trading system project, day by day.

GitHub: [UdayKiranJadi/Kairos](https://github.com/UdayKiranJadi/Kairos)