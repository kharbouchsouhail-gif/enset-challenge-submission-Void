import yaml
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the root project directory is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

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