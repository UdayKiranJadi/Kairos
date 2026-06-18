"""
universe.py — Tradeable symbol universe for Kairos.

WHY A SEPARATE FILE:
The trading universe is a business decision, not a config value.
It changes infrequently but deliberately — you might remove a stock
because it's being acquired, or add one because liquidity increased.
Keeping it here makes it easy to find and change without touching
any other file.

TIERS:
  NASDAQ_100     — full index, 100 symbols
  UNIVERSE_20    — top 20 by liquidity + news coverage (start here)
  UNIVERSE_5     — minimum viable set for testing

WHY START WITH 20 NOT 100:
Each symbol needs a trained LogReg model artifact.
Each symbol adds ~2s to the cycle (FinBERT + feature build).
With parallel processing, 20 symbols runs in ~3s per cycle.
100 symbols would need a GPU for FinBERT and a larger DB.
Scale up after the system is proven profitable on 20.
"""

NASDAQ_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "ASML", "AMD", "CSCO", "ADBE", "PEP", "QCOM", "INTU",
    "AMAT", "TXN", "AMGN", "ISRG", "MU", "LRCX", "BKNG", "MELI", "REGN",
    "ADI", "KLAC", "PANW", "CRWD", "FTNT", "SNPS", "CDNS", "MRVL", "CEG",
    "ORLY", "ABNB", "MNST", "WDAY", "PCAR", "CTAS", "PAYX", "AZN", "MAR",
    "CPRT", "ROST", "DXCM", "ODFL", "KDP", "IDXX", "FAST", "EA", "CTSH",
    "BIIB", "DLTR", "EXC", "VRSK", "TEAM", "GEHC", "XEL", "ANSS", "GILD",
    "ON", "DDOG", "ZS", "ALGN", "WBD", "ILMN", "SIRI",
]

# Top 20 — highest liquidity, most news, models already trainable
UNIVERSE_20 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AVGO", "COST", "NFLX",
    "AMD",  "QCOM", "ADBE", "TXN",  "AMGN",
    "INTU", "AMAT", "MU",   "LRCX", "BKNG",
]

# Minimum set for testing — 5 most liquid
UNIVERSE_5 = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

# Current active universe — change this one line to switch
ACTIVE_UNIVERSE = UNIVERSE_20