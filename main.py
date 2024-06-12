import pandas as pd 
import mlflow
from models.recommender import RecommenderSystem, ContentBasedRecommender, tune_hyperparameters 
from surprise import Dataset, Reader, SVD, KNNBasic, SlopeOne, CoClustering
from surprise.model_selection import train_test_split



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Recommender System Comparison")

# Load and preprocess data

def load_and_preprocess_data(file_path, sample_frac=0.001):
    df_full = pd.read_csv(file_path)
    df_sample = df_full.sample(frac=sample_frac)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_sample[['user_id', 'parent_asin', 'rating']], reader)
    return data

file_path = "./data/datasets_amazon/Automotive_5_core.csv"
data = load_and_preprocess_data(file_path)



def evaluate_models(data, models, param_grids):
    results = {}
    for model_name, (model_class, param_grid) in models.items():
        best_params = tune_hyperparameters(data, model_class, param_grid)
        model = model_class(**best_params)
        recommender = RecommenderSystem(data, model, model_name)
        rmse, mae = recommender.train_model()
        results[model_name] = {"rmse": rmse, "mae": mae, "best_params": best_params}
    return results

models = {
    "SVD": (SVD, {'n_factors': [50, 100, 200], 'reg_all': [0.02, 0.05]}),
    "KNNBasic": (KNNBasic, {'k': [20, 40, 60], 'sim_options': {'name': ['cosine', 'msd'], 'user_based': [False, True]}}),
    "SlopeOne": (SlopeOne, {}),
    "CoClustering": (CoClustering, {'n_cltr_u': [3, 5], 'n_cltr_i': [3, 5], 'n_epochs': [20, 40]})
}

results = evaluate_models(data, models, param_grids=None)


#results = evaluate_models(data, models)
print(results)

# Train Content-Based model
content_based_recommender = ContentBasedRecommender(data)
content_based_results = content_based_recommender.train_model()
print("Content-Based Results:", content_based_results)