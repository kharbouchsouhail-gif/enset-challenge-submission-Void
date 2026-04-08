import logging
import os
from pathlib import Path
from app.tools.scanner import DirectoryScannerTool
from app.tools.mri_analyzer import MRIAnalyzerTool
from app.tools.report_gen import LLMReportGeneratorTool
from app.tools.grad_cam_tool import GradCAMTool
from app.services.email_service import EmailService
from app.services.logger import setup_logger

logger = setup_logger(__name__)

class MedicalAIAgent:
    def __init__(self, config: dict):
        self.config = config
        
        # 1. Outils de base
        self.scanner = DirectoryScannerTool()
        self.analyzer = MRIAnalyzerTool(weights_path=config["model"]["weights_path"])
        self.report_generator = LLMReportGeneratorTool(llm_config=config.get("llm", {}))
        
        # 2. Service de notification
        self.email_service = EmailService(
            smtp_server=config["email"]["smtp_server"],
            smtp_port=config["email"]["smtp_port"],
            doctor_email=config["email"]["doctor_email"]
        )

        # 3. Initialisation de Grad-CAM (Correction du chemin absolu)
        raw_out_dir = config.get("gradcam", {}).get("output_dir", "reports/gradcam")
        gradcam_out = os.path.abspath(raw_out_dir) # Force le chemin à partir de la racine
        os.makedirs(gradcam_out, exist_ok=True)
        
        logger.info(f"📁 Dossier cible pour les images Grad-CAM : {gradcam_out}")

        self.gradcam_tool = GradCAMTool(
            weights_path=config["model"]["weights_path"],
            output_dir=gradcam_out,
            target_channel=config.get("gradcam", {}).get("target_channel", 1),
            colormap=config.get("gradcam", {}).get("colormap", "inferno"),
            save_nifti=config.get("gradcam", {}).get("save_nifti", True),
            save_gif=True,       # Active la visualisation 3D GIF
            save_html_3d=True    # Active la visualisation 3D Web
        )

    def run(self):
        data_dir = self.config["data_dir"]
        logger.info(f"🚀 Medical AI Agent Started. Scanning: {data_dir}")

        patients = self.scanner.execute(data_dir)
        if not patients:
            logger.info("No patients found.")
            return

        for patient in patients:
            name   = patient["patient_name"]
            images = patient["image_paths"]
            logger.info(f"--- Processing Patient: {name} ---")

            # 1. Analyse quantitative
            findings = self.analyzer.execute(images)

            # 2. Visualisation qualitative (Grad-CAM + 3D)
            for img_path in images:
                try:
                    logger.info(f"🧠 Generating Grad-CAM and 3D Models for {name}...")
                    cam_result = self.gradcam_tool.execute(
                        img_path,
                        patient_name=name,
                        planes=["axial", "coronal", "sagittal"],
                        n_slices=6,
                    )
                    logger.info("✅ Visualisations sauvegardées avec succès !")
                except Exception as e:
                    logger.error(f"❌ Grad-CAM failed for {img_path}: {e}")

            # 3. Notification technique
            raw_content = self._format_raw_findings(name, findings)
            logger.info(f"Sending raw technical findings for {name}...")
            self.email_service.send_report(f"{name} (RAW MONAI DATA)", raw_content)

            # 4. Génération du rapport clinique
            report = self.report_generator.execute(name, findings)

            # 5. Envoi du rapport final au médecin
            logger.info(f"Sending clinical report for {name}...")
            self.email_service.send_report(f"Clinical Report: {name}", report)

        logger.info("✅ Medical AI Agent finished pipeline execution.")

    def _format_raw_findings(self, patient_name: str, findings: list) -> str:
        summary  = f"TECHNICAL ANALYSIS DATA (MONAI) - PATIENT: {patient_name}\n"
        summary += "=" * 50 + "\n\n"
        for i, res in enumerate(findings):
            summary += f"SCAN #{i+1}:\n"
            summary += f" - Tumor Detected: {'YES' if res.get('tumor_detected') else 'NO'}\n"
            summary += f" - Voxel Volume: {res.get('tumor_voxel_volume')}\n"
            summary += f" - Severity: {res.get('severity')}\n"
            summary += f" - Status: {res.get('status', 'Analyzed')}\n"
            summary += "-" * 30 + "\n"
        return summary