# KKBOX User Churn Prediction

This project builds and evaluates machine learning models to predict user churn for a subscription-based music streaming platform. It leverages transactional records, user behavior logs, and demographic features from the KKBOX dataset, with additional Semi-Supervised Learning techniques. This guide explains how to set up and use the churn_prediction repository on any local machine.

## 1. Data Overview

The `data/` directory contains all data used and generated:

## data_raw/
Raw data files downloaded from the [Kaggle KKBOX Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge). These files are **not modified** and serve as the initial data source.

Typical Files:
- `train.csv`, `train_v2.csv`: Labeled user data for training.
- `sample_submission_v2.csv`: Unlabeled data used in SSL (semi-supervised learning).
- `user_logs.csv`, `user_logs_v2.csv`: Daily usage records.
- `transactions.csv`, `transactions_v2.csv`: Payment and subscription information.
- `members.csv`, `members_v3.csv`: User demographic details.

## data_intermediate/
Partially processed datasets after initial cleaning, joining, or filtering. Used to facilitate feature engineering.

## data_feature/
Constructed user-level features such as aggregated listening behavior, subscription patterns, and login activity.

Examples:
- `jan_log_user_features.csv`: Aggregated features for January logs.
- `transaction_features.csv`: Aggregated features derived from transaction history.

## data_final/
Final datasets used for model training and evaluation.

- `labeled_data.csv`: Feature matrix with known churn labels.
- `unlabeled_data.csv`: Feature matrix without labels (used in semi-supervised learning).
Stored in [Churn Prediction Dataset (Processed)](https://www.kaggle.com/datasets/hannahzz1116/churn-prediction) for convenience.

## 2. Jupyter Notebooks Overview

The `notebooks/` directory contains all development and evaluation workflows:

- **`Multilayer_Perceptron.ipynb`**  
  Implements a **Multi-Layer Perceptron (MLP)** for churn prediction

- **`Semi-supervised_Learning.ipynb`**  
  Integrates a **Semi-Supervised Learning (SSL)** strategy using pseudo-labeling.

- **`Benchmark_Models.ipynb`**  
  Trains two classic baselines:  
  - **GLM (Logistic Regression)** for interpretability  
  - **LightGBM** for scalable performance

- **`Model_Comparison.ipynb`**  
  Compares the performance of all models using:
  - Evaluation metrics (F1, AUC, Precision, Recall)  
  - Visualizations (ROC/PR curves, SHAP analysis)

## 3. Environment Setup

### Step 1: Clone the Repository
```
git clone https://github.com/MLecon/project-Hannahzihan
cd churn_prediction
```
### Step 2: Create a Virtual Environment
```
python -m venv churn_env
source churn_env/bin/activate    # or: churn_env\Scripts\activate (Windows)
```
### Step 3: Install Dependencies
```
pip install -r requirements.txt
```
```
pip install -e .
```
