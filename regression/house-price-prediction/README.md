# 🏡 House Price Prediction

My first Machine Learning project built with Python and Scikit-learn.

This project is part of my journey into Machine Learning. My goal is to learn how to work with real-world datasets, train predictive models, evaluate their performance, and understand the reasoning behind each step of the machine learning workflow.

Rather than simply using existing code, I want to understand how each component works and how data can be used to make meaningful predictions.

## 🎯 Project Objective

The main objective of this project is to build a Machine Learning model that predicts California housing values based on information about housing areas, household income, population, and other related features.

This project uses **Linear Regression**, a fundamental supervised learning algorithm, to explore the relationship between input features and a numerical target.

## 📊 Dataset

The project uses the California Housing dataset provided by Scikit-learn.

- **Source:** [Scikit-learn California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- **Number of samples:** 20,640
- **Number of input features:** 8
- **Target:** Median house value
- **Problem type:** Supervised Learning — Regression

The dataset is based on California census data from 1990. Therefore, its predictions should not be interpreted as estimates of current housing prices.

## 🛠️ Technologies

- **Python** — Main programming language
- **Pandas** — Data inspection and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Dataset loading, model training, data splitting, and evaluation

## ⚙️ Project Workflow

The project follows the main steps of a typical Machine Learning workflow:

1. Load the dataset and explore its structure.
2. Separate input features from the target variable.
3. Split the data into training and testing sets.
4. Train a Linear Regression model using the training data.
5. Generate predictions for previously unseen test samples.
6. Evaluate the model using MAE, RMSE, and R².
7. Compare model variations to understand how feature selection affects performance.

## 📈 Model Evaluation

The model will be evaluated using the following metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more strongly.
- **R² Score:** Indicates how well the model explains the variation in the target values.

Actual evaluation results will be added after the model has been trained and tested.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd machine-learning-projects/regression/house-price-prediction
```

Replace `YOUR_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python lesson_01.py
```

The California Housing dataset is loaded through Scikit-learn. An internet connection may be required the first time the dataset is downloaded.

## 🌱 Learning Goals

Through this project, I aim to develop a practical understanding of:

- The difference between features and target variables
- Training data versus testing data
- How Linear Regression learns from examples
- How to evaluate and interpret model performance
- Why data leakage must be avoided
- How to compare models and improve a machine learning workflow
- How to organize and document Python projects on GitHub

## 🔮 Future Improvements

Planned improvements include:

- Comparing models that use different feature combinations
- Visualizing actual versus predicted values
- Experimenting with additional regression algorithms
- Improving the evaluation and documentation of results
- Exploring feature engineering and model optimization

## 👨‍💻 About This Project

This is my first step toward building practical Machine Learning skills. I created this project to learn by doing, understand the fundamentals, and gradually progress toward more advanced data science and Machine Learning projects.

The repository will evolve as I learn new concepts and apply them to real datasets.

**Author:** Saman Hooti

---

*Learning Machine Learning one project at a time.*

