from pathlib import Path
from typing import Any, Dict, List
from app.core.base_tool import BaseTool
from app.services.logger import setup_logger

logger = setup_logger(__name__)

class DirectoryScannerTool(BaseTool):
    """Scans directories to find patient NIfTI files."""
    
    def execute(self, root_dir: str) -> List[Dict[str, Any]]:
        patients_data = []
        root_path = Path(root_dir)
        
        if not root_path.exists():
            logger.error(f"Root directory {root_dir} does not exist.")
            return patients_data

        for patient_folder in root_path.iterdir():
            if patient_folder.is_dir():
                mri_files = list(patient_folder.glob("*.nii")) + list(patient_folder.glob("*.nii.gz"))
                if mri_files:
                    patients_data.append({
                        "patient_name": patient_folder.name,
                        "image_paths": [str(f) for f in mri_files]
                    })
                    logger.info(f"Found {len(mri_files)} MRI(s) for patient: {patient_folder.name}")
                    
        return patients_data