# models/data_loader.py

import pandas as pd
from surprise import Dataset, Reader

def load_and_preprocess_data(file_path, sample_frac=0.001):
    df_full = pd.read_csv(file_path)
    df_sample = df_full.sample(frac=sample_frac)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_sample[['user_id', 'parent_asin', 'rating']], reader)
    return data
