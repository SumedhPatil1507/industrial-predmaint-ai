"""Slack + Email alert system."""
import logging
import asyncio
from backend.config import get_settings
from backend.database import insert_alert

logger = logging.getLogger(__name__)


async def send_slack_alert(message: str, asset_tag: str = ""):
    s = get_settings()
    if not s.slack_bot_token:
        logger.warning("Slack token not configured.")
        return False
    try:
        from slack_sdk.web.async_client import AsyncWebClient
        client = AsyncWebClient(token=s.slack_bot_token)
        await client.chat_postMessage(
            channel=s.slack_channel,
            text=f"🚨 *PredMaint Alert* | Asset: `{asset_tag}`\n{message}",
        )
        await insert_alert({"asset_tag": asset_tag, "channel": "slack",
                            "message": message, "status": "sent"})
        return True
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        await insert_alert({"asset_tag": asset_tag, "channel": "slack",
                            "message": message, "status": f"failed: {e}"})
        return False


async def send_email_alert(subject: str, body: str, asset_tag: str = ""):
    s = get_settings()
    if not s.smtp_user or not s.alert_email_to:
        logger.warning("SMTP not configured.")
        return False
    try:
        import aiosmtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "html")
        msg["Subject"] = subject
        msg["From"] = s.smtp_user
        msg["To"] = s.alert_email_to
        await aiosmtplib.send(
            msg,
            hostname=s.smtp_host,
            port=s.smtp_port,
            username=s.smtp_user,
            password=s.smtp_password,
            start_tls=True,
        )
        await insert_alert({"asset_tag": asset_tag, "channel": "email",
                            "message": subject, "status": "sent"})
        return True
    except Exception as e:
        logger.error(f"Email alert failed: {e}")
        return False


async def fire_breakdown_alert(asset_tag: str, machine_type: str,
                               probability: float, risk_level: str):
    msg = (
        f"⚠️ Breakdown predicted for *{asset_tag}* ({machine_type})\n"
        f"Probability: `{probability:.1%}` | Risk: `{risk_level}`\n"
        f"Immediate inspection recommended."
    )
    html_body = f"""
    <h2>🚨 Breakdown Alert</h2>
    <p><b>Asset:</b> {asset_tag} ({machine_type})</p>
    <p><b>Probability:</b> {probability:.1%}</p>
    <p><b>Risk Level:</b> {risk_level}</p>
    <p>Please schedule immediate inspection.</p>
    """
    await asyncio.gather(
        send_slack_alert(msg, asset_tag),
        send_email_alert(f"[PredMaint] Breakdown Alert – {asset_tag}", html_body, asset_tag),
        return_exceptions=True,
    )
