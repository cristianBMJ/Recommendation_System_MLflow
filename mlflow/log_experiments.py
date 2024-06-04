import pandas as pd 

import mlflow

# Enable server
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

import mlflow.sklearn
from surprise import Dataset, Reader, SVD , KNNBasic   
# use it others datasets
from surprise.model_selection import train_test_split
from surprise import accuracy

import mlflow 

# demo dataset
df = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'item_id': [1, 2, 3, 4, 5],
    'rating': [5, 3, 4, 2, 1]
}) 


# Load the demo dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['customer_id', 'item_id', 'rating']], reader) # load dataset

# Split the dataset into training and testing
trainset, testset = train_test_split(data, test_size=0.25)


# Amazon's datasets

# full datasets: Automative Category 
file = "./data/datasets_amazon/Automotive_5_core.csv"
df_full = pd.read_csv(file)


# Absolute Timestamp Slitting
file_train = "/home/cris/Data_Science/ML_engineer/workspace/Rec_System_MLflow/data/datasets_amazon/Automotive.train_5core_ATS.csv.gz"
df_train_full = pd.read_csv(file_train)

file_valid = "./data/datasets_amazon/Automotive.valid_5core_ATS.csv.gz" 
df_valid_full = pd.read_csv(file_valid)

file_test = "./data/datasets_amazon/Automotive.test_5core_ATS.csv.gz" # path Valid
df_test_full = pd.read_csv(file_test)



# now work amazon's datasets  
df = df_full.sample(frac=0.001)

df_train_sample = df_train_full.sample(frac=0.001)
df_valid_sample = df_valid_full.sample(frac=0.001)

print(df_train_sample.info() )

df_train = Dataset.load_from_df(df_train_sample[['user_id', 'parent_asin', 'rating']], reader)
df_valid = Dataset.load_from_df(df_valid_sample[['user_id', 'parent_asin', 'rating']], reader)

# Load datasets into Surprise format
data = Dataset.load_from_df(df[['user_id', 'parent_asin', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)




# Recommender Systems Models RS

# SVD Model
model_svd = SVD()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Recommender System")

with mlflow.start_run(run_name='RS_SVD'):
    # Log parameters
    # Train and log model
    model_svd.fit(trainset)

    # Log metrics
    predictions = model_svd.test(testset)
    
    print( 'RMSE SVD:')
    rmse = accuracy.rmse(predictions)

    mlflow.set_tag("model_name", "SVD")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("algorithm", "SVD")
    mlflow.log_param("test_size", 0.25)
    mlflow.sklearn.log_model(model_svd, "model")


## KNN Model 
sim_options = {
    'name': 'cosine',   # Cosine similarity
    'user_based': True  # User-based collaborative filtering
}

# Define the KNNBasic algorithm
model_knn = KNNBasic(sim_options=sim_options)



with mlflow.start_run(run_name='RS_KnnBase'): # very slow to large datasets


    # Train and log model
    model_knn.fit(trainset)

    # Log parameters, metrics and model
    predictions = model_knn.test(testset)
    print( 'RMSE KnnBase:')
    rmse = accuracy.rmse(predictions)

    mlflow.set_tag("model_name", "SVD")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("algorithm", "SVD")
    mlflow.log_param("test_size", 0.25)
    mlflow.sklearn.log_model( model_knn, "model")


# Implementation ATS not compatible with SVD directly



