import joblib
from surprise import Dataset, Reader, SVD # see dataset into surprise or use it others

algo =SVD()

joblib.dump(algo, 'recommender_model.pkl')



