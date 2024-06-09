import pandas as pd 
import pandas as pd 
import mlflow
import mlflow.sklearn
from surprise import Dataset, Reader, SVD, KNNBasic   
from surprise.model_selection import train_test_split
from surprise import accuracy


# Enable server
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# Amazon's datasets
file = "./data/datasets_amazon/Automotive_5_core.csv"
df_full = pd.read_csv(file)
file_train = "/home/cris/Data_Science/ML_engineer/workspace/Rec_System_MLflow/data/datasets_amazon/Automotive.train_5core_ATS.csv.gz"
df_train_full = pd.read_csv(file_train)
file_valid = "./data/datasets_amazon/Automotive.valid_5core_ATS.csv.gz" 
df_valid_full = pd.read_csv(file_valid)
file_test = "./data/datasets_amazon/Automotive.test_5core_ATS.csv.gz"
df_test_full = pd.read_csv(file_test)

reader = Reader(rating_scale=(1, 5))

# Sample datasets
df = df_full.sample(frac=0.001)
df_train_sample = df_train_full.sample(frac=0.001)
df_valid_sample = df_valid_full.sample(frac=0.001)

# Load datasets into Surprise format
data = Dataset.load_from_df(df[['user_id', 'parent_asin', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)
df_train = Dataset.load_from_df(df_train_sample[['user_id', 'parent_asin', 'rating']], reader)
df_valid = Dataset.load_from_df(df_valid_sample[['user_id', 'parent_asin', 'rating']], reader)


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Recommender System: SVD, KnnBasic, NormalPredictor")

# SVD Model
model_svd = SVD()
with mlflow.start_run(run_name='RS_SVD'):
    model_svd.fit(trainset)
    predictions = model_svd.test(testset)
    print( 'RMSE SVD:')
    rmse = accuracy.rmse(predictions)
    mlflow.set_tag("model_name", "SVD")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("algorithm", "SVD")
    mlflow.log_param("test_size", 0.25)
    mlflow.sklearn.log_model(model_svd, "model")

# KNN Model
sim_options = {
    'name': 'cosine',
    'user_based': True
}
model_knn = KNNBasic(sim_options=sim_options)
with mlflow.start_run(run_name='RS_KnnBase'):
    model_knn.fit(trainset)
    predictions = model_knn.test(testset)
    print( 'RMSE KnnBasic:')
    rmse = accuracy.rmse(predictions)
    mlflow.set_tag("model_name", "KNNBasic")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("algorithm", "KNNBasic")
    mlflow.log_param("test_size", 0.25)
    mlflow.sklearn.log_model(model_knn, "model")


from surprise import NormalPredictor

# Content-Based Filtering Model, NormalPredictor
model_content_based = NormalPredictor()  # Example of a simple content-based filtering model

with mlflow.start_run(run_name='RS_ContentBased'):
    model_content_based.fit(trainset)
    predictions = model_content_based.test(testset)
    print( 'RMSE NormalProdictor:')

    rmse = accuracy.rmse(predictions)
    
    mlflow.set_tag("model_name", "Content-Based")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("algorithm", "Content-Based")
    mlflow.log_param("test_size", 0.25)
    mlflow.sklearn.log_model(model_content_based, "model")


# Implementation ATS not compatible with SVD directly


# from lightfm import LightFM
# from lightfm.evaluation import auc_score

# # LightFM Model
# model_lightfm = LightFM(loss='warp')  # Example of using LightFM with warp loss

# # Convert Surprise trainset to LightFM format
# from lightfm.data import Dataset
# from lightfm.data import Interactions

# dataset = Dataset()
# dataset.fit(df_train_sample['user_id'].unique(), df_train_sample['parent_asin'].unique())
# interactions, weights = dataset.build_interactions([(x[0], x[1], x[2]) for x in trainset])

# with mlflow.start_run(run_name='RS_LightFM'):
#     model_lightfm.fit(interactions)
#     auc = auc_score(model_lightfm, interactions).mean()
    
#     mlflow.set_tag("model_name", "LightFM")
#     mlflow.log_metric("auc", auc)
#     mlflow.log_param("algorithm", "LightFM")
#     mlflow.log_param("loss_function", "warp")
#     # Log the LightFM model (if supported by MLflow)