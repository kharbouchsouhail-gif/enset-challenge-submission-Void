import yaml
import os
import logging

# FIX: removed module-level logging.basicConfig() call.
# Logging configuration is handled exclusively by logger.py / setup_logger().
# A basicConfig() here would fire at import time and override the proper setup.


def load_config(file_path: str = "config.yaml") -> dict:
    """
    Load parameters from the config.yaml file.
    Returns an empty dict on error so the caller can decide how to handle it.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logging.info("config.yaml loaded successfully.")
            return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {file_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"YAML syntax error in config file: {e}")
        return {}


def ensure_directories_exist(directories: list):
    """
    Ensure that required working directories exist, creating them if needed.
    Call this once at startup before any file I/O.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory verified/created: {directory}")