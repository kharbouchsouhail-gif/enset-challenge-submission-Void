import smtplib
import os
from email.message import EmailMessage
from app.services.logger import setup_logger

logger = setup_logger(__name__)

class EmailService:
    """Handles sending automated emails with the medical reports."""
    
    def __init__(self, smtp_server: str, smtp_port: int, doctor_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.doctor_email = doctor_email
        self.sender_email = os.getenv("SMTP_USER")
        self.sender_password = os.getenv("SMTP_PASSWORD")

    def send_report(self, patient_name: str, report_content: str):
        if not self.sender_email or not self.sender_password:
            logger.error("SMTP credentials missing in .env file. Skipping email.")
            return

        msg = EmailMessage()
        msg.set_content(report_content)
        msg["Subject"] = f"Urgent: MRI AI Report - {patient_name}"
        msg["From"] = self.sender_email
        msg["To"] = self.doctor_email

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            logger.info(f"Successfully sent report for {patient_name} to {self.doctor_email}")
        except Exception as e:
            logger.error(f"Failed to send email for {patient_name}: {str(e)}")