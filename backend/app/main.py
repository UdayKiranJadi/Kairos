from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS
from sqlalchemy import text

from app.api.routes_decisions import router as decisions_router
from app.api.routes_features import router as features_router
from app.api.routes_market_data import router as market_data_router
from app.api.routes_predictions import router as predictions_router
from app.api.routes_dashboard import router as dashboard_router # <-- ADD THIS
from app.db.session import AsyncSessionLocal

app = FastAPI(title="TradeOps AI")

# --- ADD CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], # React local ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_data_router)
app.include_router(features_router)
app.include_router(predictions_router)
app.include_router(decisions_router)
app.include_router(dashboard_router) # <-- ADD THIS

# ... rest of your main.py (health checks, etc.)