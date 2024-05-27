from surprise import Dataset, Reader, SVD # see dataset into surprise or use it others
from surprise.model_selection import train_test_split
from surprise import accuracy

import pandas as pd 

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

# Build the SVD model
algo = SVD()

# Train the model
algo.fit(trainset)

# Evaluate the model
predictions = algo.test(testset)
accuracy.rmse(predictions)
