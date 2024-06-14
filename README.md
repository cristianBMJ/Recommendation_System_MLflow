# Recommender Systems
Simple Recommendation System using MLflow 

# 🐍 Python Requirements

```bash 
poetry install 
```
tasK: Add also requeriments 

# Recommender System Project
## Overview 

This project aims to develop a modular recommender system using multiple algorithms. It includes data loading and preprocessing, model training, and evaluation using various collaborative and content-based filtering techniques.

**#Table of Contents**
- [Overview](#overview)
- [Overview](#Project Structure)
- [Setup](#Setup and Installation)
- [Usage](#Usage)
- [Models](#Models)
- [Evaluation](#Evaluation)
- License

## Project Structure

```bash 
├── config
│   └── config.yaml          # Configuration file
├── models
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── evaluator.py         # Model evaluation
│   └── recommender.py       # Content-based recommender
├── mlflow.db                # MLflow tracking database
├── main.py                  # Main script to run the project
├── pyproject.toml           # Poetry configuration file
└── README.md                # Project documentation
```
## Setup and Installation

1- Clone Repository:

```bash 
git clone https://github.com/your-username/recommender-system.git
cd recommender-system
```

2- Install Poetry if you don't have it:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3- Install the dependencies:

```bash
poetry install
```

4- Activate the virtual environment:

```bash
poetry shell
```

## Usage


## Models

This project implements the following recommendation algorithms:

SVD (Singular Value Decomposition)
KNNBasic (k-Nearest Neighbors)
SlopeOne
CoClustering
Additionally, a content-based recommender system is included.

## Evaluation

Model performance is evaluated using the functions provided in models/evaluator.py. The results are printed to the console and can be tracked using MLflow.

## Troubleshooting
