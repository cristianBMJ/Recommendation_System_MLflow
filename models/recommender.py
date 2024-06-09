from surprise import Dataset, Reader, SVD, KNNBasic, NormalPredictor
from surprise.model_selection import train_test_split
from surprise import accuracy
import mlflow

class RecommenderSystem:
    def __init__(self, data, model, model_name):
        self.data = data
        self.model = model
        self.model_name = model_name
        self.trainset, self.testset = train_test_split(self.data, test_size=0.25)


    def train_model(self):
        with mlflow.start_run(run_name=f'RS_{self.model_name}'):
            self.model.fit(self.trainset)
            predictions = self.model.test(self.testset)
            rmse = accuracy.rmse(predictions)
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_param("algorithm", self.model_name)
            mlflow.log_param("test_size", 0.25)
            mlflow.sklearn.log_model(self.model, "model")

class ContentBasedRecommender(RecommenderSystem):
    def __init__(self, data):
        model = NormalPredictor()
        super().__init__(data, model, "ContentBased")