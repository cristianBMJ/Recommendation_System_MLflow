import mlflow.sklearn

# Load the SVD model
model_svd = mlflow.sklearn.load_model("runs:/<SVD_RUN_ID>/model")

# Load the KNNBasic model
model_knn = mlflow.sklearn.load_model("runs:/<KNN_RUN_ID>/model")

# Load the Content-Based model
model_content_based = mlflow.sklearn.load_model("runs:/<CONTENT_BASED_RUN_ID>/model")

# Use the loaded models for making predictions or serving recommendations
# Example: Make predictions using the models
# prediction_svd = model_svd.predict(...)
# prediction_knn = model_knn.predict(...)
# prediction_content_based = model_content_based.predict(...)