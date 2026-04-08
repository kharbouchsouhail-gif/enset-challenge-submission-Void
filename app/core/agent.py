from app.tools.scanner import DirectoryScannerTool
from app.tools.mri_analyzer import MRIAnalyzerTool
from app.tools.report_gen import LLMReportGeneratorTool
from app.services.email_service import EmailService
from app.services.logger import setup_logger
import logging

logger = setup_logger(__name__)

class MedicalAIAgent:
    def __init__(self, config: dict):
        self.config = config
        self.scanner = DirectoryScannerTool()
        self.analyzer = MRIAnalyzerTool(weights_path=config["model"]["weights_path"])
        self.report_generator = LLMReportGeneratorTool(llm_config=config.get("llm", {}))        
        
        self.email_service = EmailService(
            smtp_server=config["email"]["smtp_server"],
            smtp_port=config["email"]["smtp_port"],
            doctor_email=config["email"]["doctor_email"]
        )

    def run(self):
        data_dir = self.config["data_dir"]
        logger.info(f"🚀 Medical AI Agent Started. Scanning: {data_dir}")
        
        patients = self.scanner.execute(data_dir)
        if not patients:
            logger.info("No patients found.")
            return

        for patient in patients:
            name = patient["patient_name"]
            images = patient["image_paths"]
            logger.info(f"--- Processing Patient: {name} ---")
            
            # 1. Analyse MONAI
            findings = self.analyzer.execute(images)
            
            # 2. Envoi des données BRUTES MONAI par email
            raw_content = self._format_raw_findings(name, findings)
            logger.info(f"Sending raw technical findings for {name}...")
            self.email_service.send_report(f"{name} (RAW MONAI DATA)", raw_content)
            
            # 3. Génération du Rapport Groq (qui contient aussi les données MONAI)
            report = self.report_generator.execute(name, findings)
            
            # 4. Envoi du Rapport Médical Final
            logger.info(f"Sending clinical report for {name}...")
            self.email_service.send_report(name, report)
            
        logger.info("✅ Medical AI Agent finished pipeline execution.")

    def _format_raw_findings(self, patient_name: str, findings: list) -> str:
        summary = f"TECHNICAL ANALYSIS DATA (MONAI) - PATIENT: {patient_name}\n"
        summary += "="*50 + "\n\n"
        for i, res in enumerate(findings):
            summary += f"SCAN #{i+1}:\n"
            summary += f" - Tumor Detected: {'YES' if res.get('tumor_detected') else 'NO'}\n"
            summary += f" - Voxel Volume: {res.get('tumor_voxel_volume')}\n"
            summary += f" - Severity: {res.get('severity')}\n"
            summary += "-"*30 + "\n"
        return summary