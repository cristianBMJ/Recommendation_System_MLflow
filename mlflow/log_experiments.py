import pandas as pd 

import mlflow
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


# Load the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['customer_id', 'item_id', 'rating']], reader) # load dataset

# Split the dataset into training and testing
trainset, testset = train_test_split(data, test_size=0.25)



## Model SVD
model_svd = SVD()
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Recommendation System")


with mlflow.start_run(run_name='Recomendation_System_SVD'):
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


## Model KNN
sim_options = {
    'name': 'cosine',   # Cosine similarity
    'user_based': True  # User-based collaborative filtering
}

# Define the KNNBasic algorithm
model_knn = KNNBasic(sim_options=sim_options)



with mlflow.start_run(run_name='Recomendation_System_KnnBase'):


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

print('\n')