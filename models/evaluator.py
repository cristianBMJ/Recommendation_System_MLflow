# models/evaluator.py

from models.recommender import RecommenderSystem, tune_hyperparameters

def evaluate_models(data, models):
    results = {}
    for model_name, (model_class, param_grid) in models.items():
        best_params = tune_hyperparameters(data, model_class, param_grid)
        model = model_class(**best_params)
        recommender = RecommenderSystem(data, model, model_name)
        rmse, mae = recommender.train_model()
        results[model_name] = {"rmse": rmse, "mae": mae, "best_params": best_params}
    return results
