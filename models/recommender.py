import mlflow
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from surprise import NormalPredictor

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
            mae = accuracy.mae(predictions)
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_param("algorithm", self.model_name)
            mlflow.log_param("test_size", 0.25)
            mlflow.sklearn.log_model(self.model, "model")
        return rmse, mae

class ContentBasedRecommender(RecommenderSystem):
    def __init__(self, data):
        model = NormalPredictor()
        super().__init__(data, model, "ContentBased")

    def train_model(self):
        with mlflow.start_run(run_name=f'RS_{self.model_name}'):
            self.model.fit(self.trainset)
            predictions = self.model.test(self.testset)
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)
            mlflow.set_tag("model_name", self.model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_param("algorithm", self.model_name)
            mlflow.log_param("test_size", 0.25)
            mlflow.sklearn.log_model(self.model, "model")
        return rmse, mae

def tune_hyperparameters(data, model_class, param_grid):
    trainset = data.build_full_trainset()
    gs = GridSearchCV(model_class, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    return gs.best_params['rmse']    

