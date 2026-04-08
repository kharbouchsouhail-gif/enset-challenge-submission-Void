from app.core.base_tool import BaseTool
# On importe le modèle que nous venons d'adapter (SegResNet)
from app.models.monai_wrapper import TumorDetectionModel 
from app.services.logger import setup_logger

logger = setup_logger(__name__)

class MRIAnalyzerTool(BaseTool):
    """
    Exécute l'analyse d'imagerie par résonance magnétique (IRM) 
    en utilisant le modèle MONAI réel.
    """
    
    def __init__(self, weights_path: str = None):
        # Initialisation du modèle avec les poids réels (.pt)
        # On définit une taille spatiale standard compatible avec BraTS (128x128x128)
        self.model = TumorDetectionModel(
            weights_path=weights_path, 
            spatial_size=(128, 128, 128)
        )

    def execute(self, image_paths: list) -> list:
        """
        Parcourt une liste de fichiers .nii et retourne les résultats 
        de segmentation IA réels.
        """
        results = []
        
        if not image_paths:
            logger.warning("Aucune image IRM à analyser.")
            return results

        for path in image_paths:
            logger.info(f"Analyse IA en cours (Vrai Modèle) : {path}")
            
            try:
                # Appel de la méthode predict adaptée au modèle SegResNet
                prediction = self.model.predict(path)
                
                # On enrichit le dictionnaire de sortie avec le chemin du fichier
                prediction["file"] = path
                
                # Log du résultat technique pour le suivi
                if prediction.get("tumor_detected"):
                    logger.info(f"✅ Tumeur détectée ! Volume: {prediction['tumor_voxel_volume']} voxels.")
                else:
                    logger.info(f"ℹ️ Aucune anomalie significative détectée sur {path}.")
                
                results.append(prediction)
                
            except Exception as e:
                logger.error(f"Erreur critique lors de l'analyse du fichier {path} : {str(e)}")
                # On ajoute un résultat d'erreur pour ne pas bloquer tout le pipeline
                results.append({
                    "file": path,
                    "tumor_detected": False,
                    "status": f"Error: {str(e)}",
                    "severity": "Unknown"
                })
                
        return results