import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# --- FIX DU CHEMIN D'EXÉCUTION ---
# Force le script à s'exécuter depuis le dossier racine "mri_agent_v2"
# Cela évite que les rapports ne se créent dans le sous-dossier "scripts/"
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
# ---------------------------------

from app.core.agent import MedicalAIAgent

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load environment variables (.env)
    load_dotenv()
    
    # Load YAML config
    config = load_config()
    
    # Instantiate and Run Agent
    agent = MedicalAIAgent(config)
    agent.run()

if __name__ == "__main__":
    main()