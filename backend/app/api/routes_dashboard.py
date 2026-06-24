from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import PortfolioSnapshot, PaperOrder, Symbol

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    # 1. Get the most recent portfolio snapshot
    snap_result = await db.execute(
        select(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp.desc()).limit(1)
    )
    latest_snapshot = snap_result.scalar_one_or_none()

    # 2. Get the 10 most recent paper orders
    orders_result = await db.execute(
        select(PaperOrder, Symbol.ticker)
        .join(Symbol, PaperOrder.symbol_id == Symbol.id)
        .order_by(PaperOrder.timestamp.desc())
        .limit(10)
    )
    
    recent_orders = []
    for order, ticker in orders_result.all():
        recent_orders.append({
            "id": order.id,
            "ticker": ticker,
            "side": order.side,
            "qty": order.quantity,
            "status": order.status,
            "price": order.submitted_price,
            "timestamp": order.timestamp.isoformat()
        })

    return {
        "portfolio": {
            "equity": latest_snapshot.equity if latest_snapshot else 0.0,
            "cash": latest_snapshot.cash if latest_snapshot else 0.0,
            "daily_pnl": latest_snapshot.daily_pnl if latest_snapshot else 0.0,
            "drawdown_pct": latest_snapshot.drawdown_pct if latest_snapshot else 0.0,
        },
        "recent_orders": recent_orders
    }


@router.get("/eval-metrics")
async def get_eval_metrics():
    """
    Returns saved training metrics for all LogReg models and the RL eval curve.
    Reads from artifact files on disk — no DB query needed.
    """
    logreg = []
    model_dir = Path("artifacts/models")
    if model_dir.exists():
        for path in sorted(model_dir.glob("*.joblib")):
            try:
                bundle = joblib.load(path)
                m = bundle.get("metrics", {})
                logreg.append({
                    "symbol":      bundle.get("symbol", path.stem.split("_")[0].upper()),
                    "accuracy":    round(m.get("accuracy", 0), 4),
                    "roc_auc":     round(m.get("roc_auc", 0), 4) if m.get("roc_auc") else None,
                    "rows":        m.get("rows", 0),
                    "train_rows":  m.get("train_rows", 0),
                    "test_rows":   m.get("test_rows", 0),
                    "horizon_min": bundle.get("horizon_minutes", 5),
                    "version":     bundle.get("model_version", "v0.1"),
                })
            except Exception:
                pass

    rl_curve = []
    rl_summary = {}
    eval_path = Path("artifacts/rl/logs/evaluations.npz")
    if eval_path.exists():
        ev = np.load(str(eval_path))
        timesteps   = ev["timesteps"].tolist()
        results     = ev["results"]      # (n_evals, n_episodes)
        ep_lengths  = ev["ep_lengths"]   # (n_evals, n_episodes)

        for i, t in enumerate(timesteps):
            ep_rewards  = results[i]
            ep_lens     = ep_lengths[i]
            mean_reward = float(ep_rewards.mean())
            win_rate    = float((ep_rewards > 0).mean())
            rl_curve.append({
                "timestep":       int(t),
                "mean_reward":    round(mean_reward, 4),
                "std_reward":     round(float(ep_rewards.std()), 4),
                "win_rate":       round(win_rate, 4),
                "mean_ep_length": round(float(ep_lens.mean()), 1),
            })

        final = rl_curve[-1] if rl_curve else {}
        rl_summary = {
            "total_timesteps": int(timesteps[-1]) if timesteps else 0,
            "n_checkpoints":   len(timesteps),
            "final_mean_reward": final.get("mean_reward"),
            "final_win_rate":    final.get("win_rate"),
            "best_mean_reward":  round(max(r["mean_reward"] for r in rl_curve), 4) if rl_curve else None,
            "best_win_rate":     round(max(r["win_rate"]    for r in rl_curve), 4) if rl_curve else None,
        }

    return {
        "logreg": logreg,
        "rl_curve": rl_curve,
        "rl_summary": rl_summary,
    }