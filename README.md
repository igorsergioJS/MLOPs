# Project 01 — MLOps: Regression with PyTorch

📋 Project description

This repository contains code and experiments for a regression task using the Inside Airbnb dataset for Rio de Janeiro. The primary objective is to improve a PyTorch-based neural network for predicting a numerical target (rental price or similar) by enhancing exploratory data analysis (EDA), feature engineering, preprocessing, and model architecture. The PyTorch model is compared against automated machine learning tools such as LazyPredict and PyCaret to benchmark results.

🎯 Objectives

- Improve EDA and feature selection/creation.
- Apply different normalization and scaling techniques (e.g., Min-Max, StandardScaler).
- Test multiple optimization algorithms (Adam, Nadam, etc.).
- Modify the neural network structure — layers, regularization, activation functions, and learning rate.
- Compare the model’s results with:
	- LazyPredict
	- PyCaret

Dataset

The dataset used is the Inside Airbnb Rio de Janeiro export (a cleaned CSV is included in this repo as `rio_iqr_3.csv`). The dataset contains listing-level features that can be used for regression tasks (price prediction or similar targets).

Files in this repository

- `lazypredict_rio.py` — experiments and benchmarking with LazyPredict.
- `pycaret_rio.py` — experiments and benchmarking with PyCaret.
- `lazypredict_rio.py` and `pycaret_rio.py` include data preparation and model comparison pipelines.
- `MLOPs_1(29_09).ipynb` — notebook with EDA, experiments and notes from the project.
- `rio_iqr_3.csv` — cleaned dataset used in the experiments.
- `lazypredict_rio.py` and other scripts contain the PyTorch model and preprocessing used for the final comparisons.

How to run (local Python environment)

1. Create and activate a virtual environment (recommended).

2. Install dependencies. At minimum, you'll need: pandas, numpy, scikit-learn, torch, lazypredict, pycaret (versions used in the experiments may vary). Create a `requirements.txt` if you want reproducible installs.

3. Run the notebooks or scripts:

- Open and run `MLOPs_1(29_09).ipynb` in Jupyter/Colab to reproduce the exploratory analysis and step-by-step experiments.
- Run the script benchmarks:

	- `python lazypredict_rio.py` — run LazyPredict experiments.
	- `python pycaret_rio.py` — run PyCaret experiments.

Notes and suggestions

- This project focuses on systematic comparison between a handcrafted PyTorch neural network and automated ML frameworks. When reproducing results, pay attention to random seeds, train/validation splits, and scaling pipelines to ensure fair comparisons.
- Consider adding a `requirements.txt` with pinned versions and a short script to train/evaluate the final PyTorch model for convenience.

License

See the `LICENSE` file in the repository for license details.

Link to the original Colab (project notes)

[Project 1 Colab notebook](https://colab.research.google.com/drive/1sMxKBGFPIAxHVgBtwd_GbpEMqd_vTK_9)


