from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "TradeOps AI"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://tradeops:tradeops@localhost:5432/tradeops"
    redis_url: str = "redis://localhost:6379/0"

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    alpaca_data_feed: str = "iex"

    trading_mode: str = "paper"
    risk_profile: str = "LEVEL_0_CONSERVATIVE"

    starting_capital: float = 10_000
    max_position_pct: float = 0.02
    max_daily_loss_pct: float = 0.005
    max_total_drawdown_pct: float = 0.02
    max_trades_per_day: int = 3
    max_open_positions: int = 1

    slack_webhook_url:      str = ""
    alert_email_from:       str = ""
    alert_email_to:         str = ""
    alert_email_password:   str = ""

    class Config:
        env_file = ".env"


settings = Settings()