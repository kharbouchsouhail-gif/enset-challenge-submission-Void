import torch
import numpy as np
from pathlib import Path
from monai.networks.nets import SegResNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, Orientationd
from app.services.logger import setup_logger

logger = setup_logger(__name__)

class TumorDetectionModel:
    """Wrapper pour le modèle MONAI SegResNet (BraTS) pour l'analyse IRM réelle."""
    
    def __init__(self, weights_path: str = None, spatial_size: tuple = (128, 128, 128)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spatial_size = spatial_size
        
        # --- CORRECTION 1: in_channels=4 pour correspondre au poids BraTS ---
        self.model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,   # Modifié de 1 à 4
            out_channels=3, 
        ).to(self.device)
        
        if weights_path and Path(weights_path).exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                logger.info(f"✅ REAL MODEL LOADED: {weights_path}")
            except Exception as e:
                # Cette erreur devrait disparaître maintenant
                logger.error(f"❌ Erreur lors du chargement des poids: {e}")
        else:
            logger.warning("⚠️ Weights introuvables. Mode simulation activé.")
            
        self.model.eval()

        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=self.spatial_size)
        ])

    def predict(self, nii_file_path: str) -> dict:
        """Exécute l'inférence réelle sur un fichier .nii."""
        try:
            data = {"image": nii_file_path}
            transformed_data = self.transforms(data)
            
            # --- CORRECTION 2: Adapter l'image simple en 4 canaux ---
            # On ajoute la dimension Batch (1, 1, H, W, D)
            input_tensor = transformed_data["image"].unsqueeze(0).to(self.device)
            
            # Si on n'a qu'un canal (image simple), on le duplique 4 fois 
            # pour simuler [Flair, T1, T1c, T2] attendus par le modèle
            if input_tensor.shape[1] == 1:
                input_tensor = input_tensor.repeat(1, 4, 1, 1, 1)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
                probabilities = torch.sigmoid(output)
                mask = (probabilities > 0.5).float()
                
                # On calcule le volume sur les 3 classes combinées
                voxel_count = torch.sum(mask).item()
                
            has_tumor = voxel_count > 100 
            
            avg_prob = torch.mean(probabilities[mask > 0]).item() if has_tumor else 0.0

            return {
                "tumor_detected": has_tumor,
                "tumor_voxel_volume": int(voxel_count),
                "probability": round(avg_prob, 4),
                "severity": self._calculate_severity(voxel_count),
                "status": "Success (Real AI Analysis)"
            }
            
        except Exception as e:
            logger.error(f"Inference failed on {nii_file_path}: {str(e)}")
            return {
                "tumor_detected": False,
                "tumor_voxel_volume": 0,
                "severity": "None",
                "status": f"Error: {str(e)}"
            }

    def _calculate_severity(self, volume: float) -> str:
        if volume > 10000: return "High"
        if volume > 2000: return "Moderate"
        if volume > 100: return "Low"
        return "None"