# backend/app/core/universe.py
NASDAQ_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "ASML", "AMD", "CSCO", "ADBE", "PEP", "QCOM", "INTU",
    "AMAT", "TXN", "AMGN", "ISRG", "MU", "LRCX", "BKNG", "MELI", "REGN",
    "ADI", "KLAC", "PANW", "CRWD", "FTNT", "SNPS", "CDNS", "MRVL", "CEG",
    "ORLY", "ABNB", "MNST", "WDAY", "PCAR", "CTAS", "PAYX", "AZN", "MAR",
    "CPRT", "ROST", "DXCM", "ODFL", "KDP", "IDXX", "FANG", "FAST", "EA",
    "CTSH", "BIIB", "DLTR", "EXC", "VRSK", "TEAM", "GEHC", "XEL", "ANSS",
    "GILD", "ON", "DDOG", "ZS", "ALGN", "WBD", "ILMN", "SIRI", "LCID"
]

# Start with these 20 — highest liquidity, most news coverage
TRADING_UNIVERSE_20 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AVGO", "COST", "NFLX",
    "AMD", "QCOM", "ADBE", "TXN", "AMGN",
    "INTU", "AMAT", "MU", "LRCX", "BKNG"
]