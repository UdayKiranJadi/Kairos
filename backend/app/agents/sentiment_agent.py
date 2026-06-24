"""
sentiment_agent.py
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from app.core.config import settings

logger = logging.getLogger(__name__)

_finbert_pipeline = None
_news_client = None
_openai_client = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT...")
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded")
        except Exception as e:
            logger.error(f"FinBERT load failed: {e}")
            _finbert_pipeline = None
    return _finbert_pipeline


def _get_news_client():
    global _news_client
    if _news_client is None:
        _news_client = NewsClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
    return _news_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            key = getattr(settings, "openai_api_key", "")
            if not key:
                return None
            _openai_client = OpenAI(api_key=key)
        except Exception as e:
            logger.warning(f"OpenAI init failed: {e}")
    return _openai_client


class SentimentAgent:

    OPENAI_THRESHOLD = 0.3
    CACHE_TTL_SECONDS = 300  # 5-min buckets — news doesn't move faster than this intraday

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._cache_ts: dict[str, datetime] = {}

    def _fetch_headlines(self, ticker: str, limit: int = 10) -> list[dict]:
        """
        Fetch headlines from Alpaca.

        Alpaca returns News Pydantic objects — access as attributes
        (.headline, .source, .created_at), NOT as dict keys.
        This method is the anti-corruption layer between
        Alpaca types and Kairos internal format.
        """
        try:
            client = _get_news_client()
            req = NewsRequest(
                symbols=ticker,
                start=datetime.now(timezone.utc) - timedelta(hours=6),
                limit=limit,
            )
            result = client.get_news(req)
            articles = result.data.get("news", [])
            return [
                {
                    "headline": article.headline,
                    "source": article.source,
                    "created": article.created_at.isoformat(),
                }
                for article in articles
                if article.headline
            ]
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")
            return []

    def _score_with_finbert(
        self,
        headlines: list[dict],
    ) -> tuple[float, list[dict]]:
        """
        Score each headline with FinBERT.

        score = P(positive) - P(negative) → range [-1, +1]
        aggregate = weighted mean, weighted by abs(score)
        so strong signals count more than neutral noise.
        """
        pipe = _get_finbert()
        if pipe is None or not headlines:
            return 0.0, []

        try:
            texts = [h["headline"] for h in headlines]
            results = pipe(texts)
            scored = []

            for i, result in enumerate(results):
                probs = {r["label"].lower(): r["score"] for r in result}
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                score = pos - neg

                if score > 0.1:
                    label = "positive"
                elif score < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                scored.append({
                    "headline": headlines[i]["headline"],
                    "score": round(float(score), 4),
                    "label": label,
                    "source": headlines[i]["source"],
                })

            if not scored:
                return 0.0, []

            scores = np.array([s["score"] for s in scored])
            weights = np.abs(scores) + 0.01
            aggregate = float(np.average(scores, weights=weights))
            return round(aggregate, 4), scored

        except Exception as e:
            logger.warning(f"FinBERT scoring failed: {e}")
            return 0.0, []

    def _reason_with_openai(
        self,
        ticker: str,
        aggregate_score: float,
        scored_headlines: list[dict],
        price_context: dict,
    ) -> dict:
        """
        Call GPT-4o-mini when FinBERT shows strong signal.

        Reasons across price action + news together.
        Returns structured JSON: bias, confidence, reasoning.
        Falls back to FinBERT-only result if OpenAI unavailable.
        """
        client = _get_openai()

        if client is None:
            if aggregate_score > 0.1:
                bias = "bullish"
            elif aggregate_score < -0.1:
                bias = "bearish"
            else:
                bias = "neutral"
            return {
                "bias": bias,
                "confidence": min(abs(aggregate_score) + 0.5, 1.0),
                "reasoning": f"FinBERT only (OpenAI not configured): {aggregate_score:+.3f}",
                "news_impact": "medium" if abs(aggregate_score) > 0.2 else "low",
            }

        headlines_text = "\n".join([
            f"  [{h['label'].upper():8s}] {h['headline']}"
            for h in scored_headlines[:5]
        ])

        prompt = f"""You are a quantitative analyst assessing intraday trading signals.

Ticker: {ticker}
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

PRICE CONTEXT:
  Current price:  ${price_context.get('close', 'N/A')}
  RSI(14):        {price_context.get('rsi_14', 'N/A')} (normalized -1 to +1)
  MACD signal:    {price_context.get('macd_signal', 'N/A')}
  Price vs VWAP:  {price_context.get('price_vs_vwap', 'N/A')}
  Volume z-score: {price_context.get('volume_zscore', 'N/A')}

RECENT HEADLINES (FinBERT aggregate: {aggregate_score:+.3f}):
{headlines_text}

Assess the intraday trading bias for the NEXT 5-15 MINUTES only.
Does the news confirm or contradict the technical picture?

Respond with ONLY valid JSON, no markdown, no other text:
{{
  "bias": "bullish" or "bearish" or "neutral",
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence max",
  "news_impact": "high" or "medium" or "low"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            logger.info(
                f"[{ticker}] GPT-4o-mini | "
                f"bias={result.get('bias')} | "
                f"conf={result.get('confidence', 0):.2f} | "
                f"{result.get('reasoning', '')[:60]}"
            )
            return result
        except Exception as e:
            logger.warning(f"OpenAI call failed for {ticker}: {e}")
            if aggregate_score > 0.1:
                bias = "bullish"
            elif aggregate_score < -0.1:
                bias = "bearish"
            else:
                bias = "neutral"
            return {
                "bias": bias,
                "confidence": min(abs(aggregate_score) + 0.5, 1.0),
                "reasoning": f"OpenAI error — FinBERT fallback: {aggregate_score:+.3f}",
                "news_impact": "low",
            }

    def _build_signal(
        self,
        ticker: str,
        price_context: dict | None = None,
    ) -> dict:
        """
        Core signal builder — synchronous.

        System design: this is the orchestrator method.
        It calls three sub-systems in sequence:
          1. News API (external, can fail)
          2. FinBERT (local, fast)
          3. OpenAI (external, expensive, conditional)

        Each sub-system has its own error handling so
        a failure in one doesn't crash the whole pipeline.
        This is called defensive programming — always assume
        external systems will fail and plan for it.
        """
        now = datetime.now(timezone.utc)

        # Cache check — avoid re-fetching within same cycle
        cached_ts = self._cache_ts.get(ticker)
        if cached_ts and (now - cached_ts).seconds < self.CACHE_TTL_SECONDS:
            cached = self._cache.get(ticker)
            if cached:
                return {**cached, "source": "cached"}

        # Step 1 — fetch headlines
        headlines = self._fetch_headlines(ticker)

        if not headlines:
            signal = {
                "ticker": ticker,
                "aggregate_score": 0.0,
                "bias": "neutral",
                "confidence": 0.5,
                "reasoning": "No recent news",
                "news_impact": "low",
                "headlines_count": 0,
                "top_headlines": [],
                "source": "no_news",
                "timestamp": now.isoformat(),
            }
            self._cache[ticker] = signal
            self._cache_ts[ticker] = now
            return signal

        # Step 2 — score with FinBERT
        aggregate_score, scored_headlines = self._score_with_finbert(headlines)

        # Step 3 — reason with OpenAI if signal is strong enough
        if abs(aggregate_score) >= self.OPENAI_THRESHOLD and price_context:
            gpt_result = self._reason_with_openai(
                ticker,
                aggregate_score,
                scored_headlines,
                price_context,
            )
            source = "finbert+gpt4o"
        else:
            if aggregate_score > 0.1:
                bias = "bullish"
            elif aggregate_score < -0.1:
                bias = "bearish"
            else:
                bias = "neutral"
            gpt_result = {
                "bias": bias,
                "confidence": min(abs(aggregate_score) + 0.5, 1.0),
                "reasoning": f"FinBERT aggregate: {aggregate_score:+.3f}",
                "news_impact": "medium" if abs(aggregate_score) > 0.2 else "low",
            }
            source = "finbert_only"

        signal = {
            "ticker": ticker,
            "aggregate_score": aggregate_score,
            "bias": gpt_result["bias"],
            "confidence": float(gpt_result["confidence"]),
            "reasoning": gpt_result["reasoning"],
            "news_impact": gpt_result.get("news_impact", "low"),
            "headlines_count": len(headlines),
            "top_headlines": scored_headlines[:3],
            "source": source,
            "timestamp": now.isoformat(),
        }

        self._cache[ticker] = signal
        self._cache_ts[ticker] = now
        return signal

    async def get_signal(
        self,
        ticker: str,
        price_context: dict | None = None,
    ) -> dict:
        """
        Async interface.
        Use this when calling from an async function directly.
        """
        return self._build_signal(ticker, price_context)

    def get_signal_sync(
        self,
        ticker: str,
        price_context: dict | None = None,
    ) -> dict:
        """
        Synchronous interface.

        The bot calls this via asyncio.to_thread() so FinBERT
        inference (CPU-bound, ~50ms) does not block the async
        event loop. This is important because blocking the event
        loop would delay ALL other async operations — the WS
        stream, DB writes, everything — for that 50ms.

        Rule: CPU-bound work → thread pool.
              I/O-bound work → async/await.
        """
        try:
            return self._build_signal(ticker, price_context)
        except Exception as e:
            logger.warning(f"get_signal_sync failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "aggregate_score": 0.0,
                "bias": "neutral",
                "confidence": 0.5,
                "reasoning": f"Sentiment unavailable: {str(e)[:50]}",
                "news_impact": "low",
                "headlines_count": 0,
                "top_headlines": [],
                "source": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }