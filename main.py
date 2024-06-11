import pandas as pd 
import mlflow
from models.recommender import RecommenderSystem, ContentBasedRecommender
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

# file_path = "./data/datasets_amazon/Automotive_5_core.csv"
# data = load_and_preprocess_data(file_path)

# # # Train SVD model
# # model_svd = SVD() # Singular Value Decomposition
# # svd_recommender = RecommenderSystem(data, model_svd, "SVD")
# # svd_recommender.train_model()

# # # Train KNN model
# # sim_options = {'name': 'cosine', 'user_based': False} # similariy between Users, False between Items
# # model_knn = KNNBasic(sim_options=sim_options)
# # knn_recommender = RecommenderSystem(data, model_knn, "KNNBasic")
# # knn_recommender.train_model()

# # # Train Content-Based model
# # content_based_recommender = ContentBasedRecommender(data)
# # content_based_recommender.train_model()



def evaluate_models(data, models):
    results = {}
    for model_name, model in models.items():
        recommender = RecommenderSystem(data, model, model_name)
        rmse, mae = recommender.train_model()
        results[model_name] = {"rmse": rmse, "mae": mae}
    return results

file_path = "./data/datasets_amazon/Automotive_5_core.csv"
data = load_and_preprocess_data(file_path)

models = {
    "SVD": SVD(),
    "KNNBasic": KNNBasic(sim_options={'name': 'cosine', 'user_based': False}),
    "SlopeOne": SlopeOne(),
    "CoClustering": CoClustering()
}

results = evaluate_models(data, models)
print(results)

# Train Content-Based model
content_based_recommender = ContentBasedRecommender(data)
content_based_results = content_based_recommender.train_model()
print("Content-Based Results:", content_based_results)