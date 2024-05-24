from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['customer_id', 'product_id', 'rating']], reader)

# Split the dataset into training and testing
trainset, testset = train_test_split(data, test_size=0.25)

# Build the SVD model
algo = SVD()

# Train the model
algo.fit(trainset)

# Evaluate the model
predictions = algo.test(testset)
accuracy.rmse(predictions)
