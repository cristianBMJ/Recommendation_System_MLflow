import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("algorithm", "SVD")
    mlflow.log_param("test_size", 0.25)

    # Train and log model
    algo.fit(trainset)
    mlflow.sklearn.log_model(algo, "model")

    # Log metrics
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    mlflow.log_metric("rmse", rmse)
