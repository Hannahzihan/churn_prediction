Project Setup Guidance for churn_prediction

This guide explains how to set up and use the churn_prediction repository on any local machine. It is intended for collaborators or for transferring the project to a new computer.

1. Repository Structure

churn_prediction/
├── setup.py
├── requirements.txt             
├── notebooks/                # Jupyter notebooks for model development and evaluation
│   ├── Benchmark_Models.ipynb.py
│   │   ├── MLP_with_SSL.ipynb     # MLP and best MLP with Semi-Supervised learning
│   │   ├── Benchmark_Models.ipynb # an interpretable logistic regression model (GLM) and the widely adopted LightGBM
│   │   └── Model_Comparison.ipynb # comparsion between four models
├── data/     
│   ├── data_raw/             # Raw data from Kaggle       
│   ├── data_intermediate/    # Intermediat data
│   ├── data_feature/         # Features extracted
│   └── data_final/           # Final data used for training
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

2. Environment Setup

Step 1: Clone the Repository

git clone <repo-url>
cd churn_prediction

Step 2: Create a Virtual Environment

python -m venv churn_env
source churn_env/bin/activate    # or: churn_env\Scripts\activate (Windows)

Step 3: Install Dependencies

pip install -r requirements.txt

Alternatively, if using as a package:

pip install -e .

3. Usage Tips

Run Notebook

Launch Jupyter Lab or Notebook and open files under /notebooks/:

jupyter lab

Importing Project Modules

Ensure your working directory is the project root, and use:

from src.data.preprocess import data_preprocessing
from src.model.MLPClassifier import MLPClassifier

If you encounter import errors in notebooks, you can append the root path manually:

import sys, os
sys.path.append(os.path.abspath(".."))  # from inside `notebooks/`

4. Common Issues

Problem: ModuleNotFoundError: No module named 'src'

Ensure you're running the notebook from the root or have updated sys.path

Problem: file:// path in requirements.txt

Replace any local path like churn-prediction @ file://... with -e . or remove entirely

5. Updating Dependencies

After modifying code or adding dependencies:

pip freeze > requirements.txt

Or manually maintain setup.py and requirements.txt.

6. Reproduce in New Machine

From a new machine:

git clone <repo-url>
cd churn_prediction
python -m venv churn_env
source churn_env/bin/activate
pip install -e .

7. Contact

If any issues arise or help is needed, please refer to the project's README or contact the maintainer.

This guidance ensures consistent setup, usage, and collaboration across environments and users.

