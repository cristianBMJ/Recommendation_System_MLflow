# main.py

import yaml
import mlflow
from models.data_loader import load_and_preprocess_data
from models.evaluator import evaluate_models
from models.recommender import ContentBasedRecommender
from surprise import SVD, KNNBasic, SlopeOne, CoClustering

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Recommender System Modularization")

data_config = config['data']
data = load_and_preprocess_data(data_config['file_path'], data_config['sample_frac'])

models_config = config['models']
models = {
    "SVD": (SVD, models_config['SVD']['param_grid']),
    "KNNBasic": (KNNBasic, models_config['KNNBasic']['param_grid']),
    "SlopeOne": (SlopeOne, models_config['SlopeOne']['param_grid']),
    "CoClustering": (CoClustering, models_config['CoClustering']['param_grid'])
}

results = evaluate_models(data, models)

print(results)

# Train Content-Based model
content_based_recommender = ContentBasedRecommender(data)
content_based_recommender.train_model()
