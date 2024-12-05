import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from surprise import Dataset, Reader
from models.data_loader import load_and_preprocess_data
from models.evaluator import evaluate_models
from models.recommender import ContentBasedRecommender, RecommenderSystem
from surprise import SVD, KNNBasic, SlopeOne, CoClustering
import yaml

mlflow.set_tracking_uri("databricks")
mlflow.login()
mlflow.set_experiment("/Users/cristianj3006@gmail.com/MLFlow_with_streamlit")

# Set up MLflow tracking URI
# mlflow.set_tracking_uri("http://localhost:5000")

# Streamlit app
st.title("Recommender System with MLflow")

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_config = config['data']
models_config = config['models']

@st.cache_data
def load_data(file_path, sample_frac):
    return load_and_preprocess_data(file_path, sample_frac)

# Load and preprocess data
st.write("## Data Loading")
file_path = st.text_input("Dataset file path", data_config['file_path'])
sample_frac = st.slider("Sample Fraction", min_value=0.01, max_value=1.00, value=data_config['sample_frac'])
data = load_data(file_path, sample_frac)

st.write("Data loaded successfully!")

# Evaluate models
st.write("## Model Evaluation")

if st.button("Evaluate Models"):
    with st.spinner("Evaluating models..."):
        models = {
            "SVD": (SVD, models_config['SVD']['param_grid']),
            "KNNBasic": (KNNBasic, models_config['KNNBasic']['param_grid']),
            "SlopeOne": (SlopeOne, models_config['SlopeOne']['param_grid']),
            "CoClustering": (CoClustering, models_config['CoClustering']['param_grid'])
        }
        results = evaluate_models(data, models)
        st.write(results)

# Train Content-Based model
st.write("## Train Content-Based Model")

if st.button("Train Content-Based Model"):
    with st.spinner("Training content-based model..."):
        content_based_recommender = ContentBasedRecommender(data)
        rmse, mae = content_based_recommender.train_model()
        st.write(f"Content-Based Model trained")
        st.write(f"RMSE: {round(rmse, 3)}")
        st.write(f"MAE: {round(mae, 3)}")

# Visualize MLflow experiments
st.write("## MLflow Experiments")

if st.button("Show MLflow UI"):
    st.markdown(f"[MLflow UI](http://localhost:5000)", unsafe_allow_html=True)
