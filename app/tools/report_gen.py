import os
import logging
from app.core.base_tool import BaseTool

PROMPT_TEMPLATE = """
You are an AI assistant helping radiologists analyze MRI scans.
Write a professional medical report for patient: {patient_name}.

--- MANDATORY TECHNICAL DATA FROM MONAI ---
- Tumor detected       : {tumor_detected}
- Tumor voxel volume   : {tumor_voxel_volume} voxels
- Neural Network Severity: {severity}
- Analysis status      : {status}
- Mock mode (simulated): {mock_mode}
--------------------------------------------

Structure the report with the following sections:
1. Patient Information
2. Technical Observations (Integrate the MONAI data above into this section)
3. AI Findings Summary
4. Clinical Recommendation

Important rules:
- You MUST mention the exact voxel volume and detection status in the Technical section.
- Always recommend validation by a licensed medical professional.
- Be precise and use professional medical terminology.
- Do not greet or introduce yourself — write the document directly.
- If mock mode is True, clearly state that results are simulated.
"""

class LLMReportGeneratorTool(BaseTool):
    def __init__(self, llm_config: dict = None):
        api_key = os.getenv("GROQ_API_KEY")
        self.mock_mode = False
        self.llm = None
        self.prompt = None
        self._human_message_cls = None

        cfg = llm_config or {}
        model_name  = cfg.get("model_name", "llama-3.3-70b-versatile")
        temperature = cfg.get("temperature", 0.2)
        max_tokens  = cfg.get("max_tokens", 1000)

        if not api_key:
            logging.warning("GROQ_API_KEY missing — Fallback mode activated.")
            self.mock_mode = True
            return

        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import PromptTemplate
            from langchain_core.messages import HumanMessage
            
            self._human_message_cls = HumanMessage
            self.llm = ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=api_key,
                max_tokens=max_tokens,
            )
            self.prompt = PromptTemplate(
                input_variables=["patient_name", "tumor_detected", "tumor_voxel_volume", "severity", "status", "mock_mode"],
                template=PROMPT_TEMPLATE,
            )
            logging.info(f"ReportGeneratorTool ready with model: {model_name}")
        except ImportError:
            self.mock_mode = True

    def execute(self, patient_name: str, analysis_results: list) -> str:
        merged = self._merge_results(analysis_results)

        if self.mock_mode:
            content = self._fallback_report(patient_name, merged)
        else:
            try:
                filled_prompt = self.prompt.format(
                    patient_name       = patient_name,
                    tumor_detected     = "YES" if merged["tumor_detected"] else "NO",
                    tumor_voxel_volume = merged["tumor_voxel_volume"],
                    severity           = merged["severity"],
                    status             = merged["status"],
                    mock_mode          = merged["mock_mode"],
                )
                response = self.llm.invoke([self._human_message_cls(content=filled_prompt)])
                content = response.content
            except Exception as e:
                logging.error(f"LLM Error: {e}")
                content = self._fallback_report(patient_name, merged)

        # Sauvegarde locale
        self._save_report_to_file(patient_name, content)
        return content

    def _save_report_to_file(self, patient_name: str, content: str):
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        filepath = os.path.join(reports_dir, f"{patient_name}_report.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"💾 Report saved to: {filepath}")

    def _merge_results(self, analysis_results: list) -> dict:
        if not analysis_results:
            return {"tumor_detected": False, "tumor_voxel_volume": 0, "severity": "None", "status": "Clear", "mock_mode": True}
        
        tumor_detected = any(r.get("tumor_detected", False) for r in analysis_results)
        total_voxels   = sum(r.get("tumor_voxel_volume", 0) for r in analysis_results)
        severities = [r.get("severity", "None") for r in analysis_results]
        order = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
        worst_severity = max(severities, key=lambda s: order.get(s, 0))

        return {
            "tumor_detected": tumor_detected,
            "tumor_voxel_volume": total_voxels,
            "severity": worst_severity,
            "status": "Action Required" if tumor_detected else "Clear",
            "mock_mode": any(r.get("mock_mode", False) for r in analysis_results),
        }

    def _fallback_report(self, patient_name: str, merged: dict) -> str:
        return f"MEDICAL REPORT — {patient_name}\n{'='*30}\nRaw MONAI Findings:\n- Detected: {merged['tumor_detected']}\n- Volume: {merged['tumor_voxel_volume']}\n- Severity: {merged['severity']}\n"