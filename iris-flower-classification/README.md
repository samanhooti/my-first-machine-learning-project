# 🌸 Iris Flower Classification with Machine Learning

A beginner-friendly machine learning project that uses Python and the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers based on their physical measurements.

This project was built as a hands-on learning experience to understand the fundamental workflow of a machine learning project, from loading a dataset to training a model and evaluating its predictions.

## 📌 Project Overview

The goal of this project is to predict the species of an Iris flower using four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model predicts one of three Iris species:

- **Setosa**
- **Versicolor**
- **Virginica**

## 📊 Dataset

This project uses the Iris dataset provided by `scikit-learn`, so no manual dataset download is required.

The dataset contains:

| Property | Value |
|---|---|
| Total samples | 150 |
| Input features | 4 |
| Target classes | 3 |
| Training samples | 120 |
| Testing samples | 30 |

The dataset is divided into training and testing sets. The model learns from 80% of the samples and is evaluated on the remaining 20%.

## 🛠️ Technologies Used

- **Python** — Main programming language
- **scikit-learn** — Dataset loading, model training, and evaluation
- **K-Nearest Neighbors (KNN)** — Classification algorithm
- **GitHub** — Project hosting and documentation

## ⚙️ How the Project Was Built

### 1. Loading the Dataset

The Iris dataset is loaded using `load_iris()` from `sklearn.datasets`.

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

Here, `X` contains the four flower measurements, while `y` contains the corresponding species labels.

### 2. Splitting the Data

The dataset is divided into training and testing sets using `train_test_split()`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- `test_size=0.2` reserves 20% of the dataset for testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` helps preserve the class proportions in both sets.

The result is 120 training samples and 30 testing samples.

### 3. Creating and Training the Model

The K-Nearest Neighbors algorithm classifies a flower based on the labels of nearby examples in the feature space.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)
```

The parameter `n_neighbors=3` means that the model considers the three nearest training samples when making a classification.

The `fit()` method trains the model using the training data.

### 4. Making Predictions

After training, the model predicts the species of the flowers in the test set.

```python
predictions = model.predict(X_test)
```

These predictions can then be compared with the actual labels to evaluate the model's performance.

### 5. Evaluating Accuracy

The accuracy score measures the proportion of test samples classified correctly.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")
```

In this experiment, the model achieved **100.00% accuracy** on the selected test set.

### 6. Analyzing the Confusion Matrix

A confusion matrix shows how the actual classes compare with the model's predictions.

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
```

The resulting matrix was:

```text
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]
```

Each row represents an actual class, and each column represents a predicted class. The diagonal values show correct predictions.

In this experiment, all 30 test samples were classified correctly, with 10 correct predictions for each species.

## 📈 Results

| Metric | Result |
|---|---:|
| Training samples | 120 |
| Testing samples | 30 |
| Test accuracy | 100.00% |
| Correct predictions | 30 |
| Incorrect predictions | 0 |

These results describe one train-test split using the specified random seed. They do not guarantee perfect performance on every unseen dataset or in real-world conditions.

## 🧠 What I Learned

Through this project, I practiced the core steps of a supervised machine learning workflow:

- Loading and exploring a dataset
- Separating features from target labels
- Splitting data into training and testing sets
- Training a classification model
- Generating predictions
- Evaluating accuracy
- Interpreting a confusion matrix
- Organizing and documenting a Python project

This project is part of my ongoing journey to strengthen my Python and machine learning skills through practical experiments.

## 🚀 How to Run the Project

### Requirements

- Python 3.10 or newer
- pip

### Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd machine-learning-projects/iris-flower-classification
```

Install the required library:

```bash
python -m pip install -r requirements.txt
```

Run the program:

```bash
python iris_classification.py
```

Replace `YOUR_REPOSITORY_URL` with the actual URL of your GitHub repository.

## 📁 Project Structure

```text
iris-flower-classification/
├── iris_classification.py
├── README.md
└── requirements.txt
```

- `iris_classification.py` — Loads the data, trains the model, makes predictions, and evaluates performance.
- `README.md` — Documents the project, methodology, and results.
- `requirements.txt` — Lists the Python dependencies.

## 🔭 Future Improvements

Possible next steps for this project include:

- Testing different values of `n_neighbors`
- Comparing KNN with other classification algorithms
- Using cross-validation for a more reliable performance estimate
- Visualizing the dataset and classification results
- Building a simple interface that predicts a flower species from user-provided measurements

## 👨‍💻 About

This is an educational project created to develop practical experience with Python and machine learning.

**Author:** Saman Hooti

*Learning by building, experimenting, and improving one project at a time.*

