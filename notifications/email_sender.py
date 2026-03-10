"""
Email Notification Module

Sends email notifications for weekly reports and critical alerts.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Sends email notifications for the trading system.

    Supports weekly performance reports and critical error alerts.

    Example:
        notifier = EmailNotifier()
        notifier.send_weekly_report(report_data)
        notifier.send_error_alert("Critical", "System failed")
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_address: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            username: SMTP username.
            password: SMTP password.
            from_address: Sender email address.
            recipients: List of recipient email addresses.
            enabled: Whether email sending is enabled.
        """
        from config.settings import email_settings

        self.smtp_host = smtp_host or email_settings.smtp_host
        self.smtp_port = smtp_port or email_settings.smtp_port
        self.username = username or email_settings.smtp_user
        self.password = password or email_settings.smtp_password
        self.from_address = from_address or email_settings.from_address
        self.recipients = recipients or (
            [r.strip() for r in email_settings.recipients.split(",") if r.strip()]
        )
        self.enabled = enabled if enabled is not None else email_settings.enabled

    def _send_email(self, subject: str, body_html: str) -> bool:
        """Send an email.

        Args:
            subject: Email subject.
            body_html: HTML email body.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            logger.debug(f"Email disabled. Would send: {subject}")
            return False

        if not self.recipients:
            logger.warning("No email recipients configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.recipients)

            # Add HTML body
            msg.attach(MIMEText(body_html, "html"))

            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(
                    self.from_address,
                    self.recipients,
                    msg.as_string(),
                )

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email '{subject}': {e}")
            return False

    def send_weekly_report(self, report: Dict) -> bool:
        """Send weekly performance report.

        Args:
            report: Report data dictionary with performance metrics.

        Returns:
            True if sent successfully.
        """
        subject = f"IDX Trading System - Weekly Report ({datetime.now().strftime('%Y-%m-%d')})"

        total_pnl = report.get("total_pnl", 0)
        total_pnl_pct = report.get("total_pnl_pct", 0)
        trades = report.get("trades_this_week", 0)
        win_rate = report.get("win_rate", 0)

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>📊 IDX Trading System - Weekly Report</h2>
            <p>Period: {report.get('start_date', 'N/A')} to {report.get('end_date', 'N/A')}</p>

            <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
                <tr style="background: #f0f0f0;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Total P&L</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if total_pnl >= 0 else 'red'};">
                        IDR {total_pnl:,.0f} ({total_pnl_pct:+.2f}%)
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Trades</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{trades}</td>
                </tr>
                <tr style="background: #f0f0f0;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Win Rate</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{win_rate:.1f}%</td>
                </tr>
            </table>

            <p style="color: #888; font-size: 12px; margin-top: 20px;">
                Generated by IDX Trading System v3.0.0
            </p>
        </body>
        </html>
        """

        return self._send_email(subject, body)

    def send_error_alert(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
    ) -> bool:
        """Send critical error alert.

        Args:
            error_type: Type of error.
            message: Error message.
            traceback: Optional traceback.

        Returns:
            True if sent successfully.
        """
        subject = f"⚠️ IDX Trading System ALERT: {error_type}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: red;">⚠️ System Alert</h2>
            <p><strong>Type:</strong> {error_type}</p>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Message:</strong> {message}</p>
            {f'<pre style="background: #f5f5f5; padding: 10px; overflow-x: auto;">{traceback}</pre>' if traceback else ''}
            <p style="color: #888; font-size: 12px; margin-top: 20px;">
                IDX Trading System v3.0.0
            </p>
        </body>
        </html>
        """

        return self._send_email(subject, body)
