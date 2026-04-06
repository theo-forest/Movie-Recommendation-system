# utils.py

import logging
import os
import json

def setup_logging(log_file='app.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_file='config.json'):
    if not os.path.exists(config_file):
        logging.error(f"Configuration file {config_file} not found.")
        return {}
    
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    logging.info("Configuration loaded successfully.")
    return config

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}.")

def load_model(model_path):
    import joblib
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} not found.")
        return None
    
    model = joblib.load(model_path)
    logging.info(f"Model loaded from {model_path}.")
    return model

def calculate_similarity(vec_a, vec_b):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0]