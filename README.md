## Project Setup Guidance for churn_prediction

This guide explains how to set up and use the churn_prediction repository on any local machine. It is intended for transferring the project to a new computer.

## 1. Repository Structure
```
churn_prediction/
├── setup.py
├── requirements.txt             
├── notebooks/                # Jupyter notebooks for model development and evaluation
│   ├── Benchmark_Models.ipynb.py
│   │   ├── MLP_with_SSL.ipynb     # MLP and best MLP with Semi-Supervised learning
│   │   ├── Benchmark_Models.ipynb # an interpretable logistic regression model (GLM) and the widely adopted LightGBM
│   │   └── Model_Comparison.ipynb # comparsion between four models
├── data/     
│   ├── data_raw/                  # Raw data from Kaggle       
│   ├── data_intermediate/         # Intermediat data
│   ├── data_feature/              # Features extracted
│   ├── data_final/                # Final data used for training
│   └── data_preprocessing.ipynb   # How to process from the original data
├── src/                      # Source code organized by module
│   ├── __init__.py
│   ├── data/                 # Data preprocessing and resampling
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   └── resample.py
│   ├── inference/            # Inference utilities
│   │   ├── __init__.py
│   │   ├── model_loading.py
│   │   └── predict.py
│   ├── model/                # Model architectures
│   │   ├── glm.py
│   │   ├── lgbm.py
│   │   ├── mlp.py
│   │   └── MLPClassifier.py
│   ├── training/             # Training loops and hyperparameter search
│   │   ├── __init__.py
│   │   └── train.py
│   │   └── tune.py
│   └── visualization/        # Plotting tools
│       ├── __init__.py
│       └── plot_curves.py
├── models/                   # All models trained
│   ├── glm_model.pkl
│   ├── lgbm_model.pkl
│   ├── mlp_model.pt
│   ├── mlp_ssl_model.pt
```

The `notebooks/` directory contains interactive development files used throughout the project:
  
- `MLP_with_SSL.ipynb`  
  Implements **a neural network (MLP)** to predict user churn and introduces a **Semi-Supervised Learning** strategy via pseudo-labeling on unlabeled data. Includes model training, evaluation, and comparison against supervised MLP.

- `Benchmark_Models.ipynb`  
  Trains and evaluates two baseline models:
  - **GLM (Logistic Regression)**: An interpretable linear model.
  - **LightGBM**: A gradient-boosted tree model known for efficiency and performance.

- `Model_Comparison.ipynb`  
  Brings together results from all models—GLM, LightGBM, MLP, and SSL-MLP—for comparison using classification metrics (Precision, Recall, F1, AUC) and visualizations (ROC/PR curves, SHAP explanations).

## 2. Environment Setup

Step 1: Clone the Repository
```
git clone https://github.com/Hannahzihan/churn_prediction.git
cd churn_prediction
```
Step 2: Create a Virtual Environment
```
python -m venv churn_env
source churn_env/bin/activate    # or: churn_env\Scripts\activate (Windows)
```
Step 3: Install Dependencies
```
pip install -r requirements.txt
```
Alternatively, if using as a package:
```
pip install -e .
```
