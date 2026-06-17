"""
alerting.py — Slack and email alerts for autonomous events.

WHY ALERTS MATTER FOR AUTONOMOUS SYSTEMS:
When no human approves trades, you need the system to
reach out to you when something important happens.
Three events always trigger alerts:

  1. Circuit breaker fires (daily loss limit hit)
  2. ATR stop-loss executes (position force-closed)
  3. Trade executed (buy or sell confirmed)

You don't need to watch the dashboard. The system finds you.

SETUP:
For Slack: create an incoming webhook at
  https://api.slack.com/messaging/webhooks
  Add SLACK_WEBHOOK_URL to your .env

For email: uses Gmail SMTP with an app password
  (not your main password — generate one at
  myaccount.google.com/apppasswords)
  Add ALERT_EMAIL_FROM, ALERT_EMAIL_TO,
  ALERT_EMAIL_PASSWORD to your .env

Both are optional — if the env vars are missing,
alerts are logged but not sent. No crash.
"""

import logging
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertLevel:
    INFO     = "info"      # trade executed
    WARNING  = "warning"   # stop-loss triggered
    CRITICAL = "critical"  # circuit breaker fired


async def send_slack(message: str, level: str = AlertLevel.INFO) -> None:
    """
    Send a message to Slack via incoming webhook.
    Silently skips if SLACK_WEBHOOK_URL is not configured.
    """
    webhook_url = getattr(settings, "slack_webhook_url", None)
    if not webhook_url:
        logger.debug("Slack not configured — skipping alert")
        return

    emoji = {"info": "⬡", "warning": "⚠️", "critical": "🛑"}.get(level, "⬡")
    color = {"info": "#185FA5", "warning": "#EF9F27", "critical": "#E24B4A"}.get(level, "#185FA5")

    payload = {
        "attachments": [
            {
                "color":  color,
                "text":   f"{emoji} *Kairos* — {message}",
                "footer": f"<!date^{int(datetime.now(timezone.utc).timestamp())}^{{date_short_pretty}} {{time}}|{datetime.now(timezone.utc).strftime('%H:%M UTC')}>",
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code != 200:
                logger.warning(f"Slack alert failed: {resp.status_code} {resp.text}")
            else:
                logger.info(f"Slack alert sent: {message[:60]}")
    except Exception as e:
        logger.warning(f"Slack alert error: {e}")


def send_email(subject: str, body: str) -> None:
    """
    Send an email alert via Gmail SMTP.
    Runs synchronously — call via asyncio.to_thread() from async code.
    Silently skips if email settings are not configured.
    """
    from_addr = getattr(settings, "alert_email_from", None)
    to_addr   = getattr(settings, "alert_email_to",   None)
    password  = getattr(settings, "alert_email_password", None)

    if not all([from_addr, to_addr, password]):
        logger.debug("Email not configured — skipping alert")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Kairos] {subject}"
    msg["From"]    = from_addr
    msg["To"]      = to_addr

    html = f"""
    <html><body style="font-family:sans-serif;color:#1a1a1a;padding:24px">
      <h2 style="color:#185FA5;margin:0 0 8px">⬡ Kairos Alert</h2>
      <p style="font-size:16px;margin:0 0 16px">{subject}</p>
      <pre style="background:#f5f5f5;padding:16px;border-radius:8px;font-size:13px">{body}</pre>
      <p style="color:#888;font-size:12px;margin-top:16px">
        Sent at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
        · Kairos autonomous trading system
      </p>
    </body></html>
    """

    msg.attach(MIMEText(body, "plain"))
    msg.attach(MIMEText(html,  "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_addr, password)
            server.sendmail(from_addr, to_addr, msg.as_string())
        logger.info(f"Email alert sent: {subject}")
    except Exception as e:
        logger.warning(f"Email alert error: {e}")


async def alert(
    message: str,
    level:   str = AlertLevel.INFO,
    subject: str | None = None,
) -> None:
    """
    Send both Slack and email alerts concurrently.
    This is the single function called everywhere in the codebase.

    Usage:
        await alert("AAPL position closed via ATR stop", level="warning")
        await alert("Circuit breaker fired — trading halted", level="critical")
        await alert("BUY AAPL 5 shares @ $213.45", level="info")
    """
    subject = subject or message[:60]

    await asyncio.gather(
        send_slack(message, level),
        asyncio.to_thread(send_email, subject, message),
        return_exceptions=True,
    )