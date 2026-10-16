# my-first-machine-learning-project
# 🌸 Iris Flower Classification

**My First Machine Learning Classification Project with Python**

A hands-on machine learning project focused on classifying Iris flower species using Python, the K-Nearest Neighbors (KNN) algorithm, and the Iris dataset provided by scikit-learn.

This project represents one of my steps toward understanding machine learning through practical coding, experimentation, and analyzing model performance.

## 📌 Project Overview

The goal of this project is to build a machine learning model that predicts the species of an Iris flower based on four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model classifies each flower into one of three species:

- **Setosa**
- **Versicolor**
- **Virginica**

Instead of manually writing rules to classify flowers, I trained a model to learn patterns from labeled examples and use those patterns to make predictions.

## 📊 Dataset

This project uses the built-in Iris dataset from `scikit-learn`. The dataset can be loaded directly in Python, so a separate download is not required.

| Dataset property | Value |
|---|---:|
| Total samples | 150 |
| Features per sample | 4 |
| Target classes | 3 |
| Training samples | 120 |
| Testing samples | 30 |

Each sample contains four numerical measurements and a label representing its flower species.

## 🛠️ Technologies and Tools

- **Python** — Writing the program and organizing the workflow
- **scikit-learn** — Loading the dataset, training the model, and evaluating predictions
- **K-Nearest Neighbors (KNN)** — The classification algorithm
- **GitHub** — Storing and documenting the project

## ⚙️ How the Project Works

The project follows a basic supervised machine learning workflow.

### 1. Loading the Dataset

I used `load_iris()` to load the dataset and access its measurements and labels.

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

- `X` contains the input features used by the model.
- `y` contains the correct species labels.

This step helped me understand how a machine learning dataset is represented in Python.

### 2. Splitting the Data

I separated the dataset into training and testing sets using `train_test_split()`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The training set contains 80% of the samples, while the testing set contains the remaining 20%.

I used `random_state=42` to make the split reproducible and `stratify=y` to preserve the class proportions in the two sets.

This step is important because a model should be evaluated on data that was not used to train it.

### 3. Creating and Training the Model

I chose the K-Nearest Neighbors algorithm for this classification task.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)
```

KNN predicts a class by examining the nearest training samples to a new example.

The parameter `n_neighbors=3` tells the model to consider the three nearest neighbors when making a prediction.

The `fit()` method trains the model using the training data.

### 4. Making Predictions

After training, I used the model to predict the species of the flowers in the test set.

```python
predictions = model.predict(X_test)
```

These predictions can be compared with the actual labels to determine how well the model performs on the test data.

### 5. Evaluating Accuracy

I used `accuracy_score()` to calculate the proportion of correct predictions.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")
```

Accuracy provides a simple way to summarize the model's performance.

### 6. Analyzing the Confusion Matrix

I also used a confusion matrix to examine the classification results for each species.

```python
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, predictions)

print(matrix)
```

The confusion matrix from my experiment was:

```text
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]
```

Rows represent the actual classes, while columns represent the predicted classes. The diagonal values indicate correct predictions.

In this experiment, all 30 test samples were classified correctly.

## 📈 Experimental Results

The following results were obtained from one train-test split using the configuration described above.

| Metric | Result |
|---|---:|
| Training samples | 120 |
| Testing samples | 30 |
| Test accuracy | 100.00% |
| Correct predictions | 30 |
| Incorrect predictions | 0 |

The model achieved **100% accuracy on this particular test set**.

This is an encouraging result, but it does not guarantee perfect performance on other datasets or real-world flower measurements. Further experiments, such as cross-validation and testing different model settings, would provide a more complete evaluation.

## 🧠 What I Learned

Building this project helped me practice several fundamental machine learning concepts.

### Python and Data Handling

- Loading a dataset with scikit-learn
- Separating input features from target labels
- Working with training and testing data
- Organizing code into clear, readable sections
- Using Python libraries to perform machine learning tasks

### Machine Learning Fundamentals

- Understanding supervised learning and classification
- Learning the basic idea behind the K-Nearest Neighbors algorithm
- Training a model with `fit()`
- Making predictions with `predict()`
- Evaluating predictions with accuracy
- Interpreting a confusion matrix

### Experimentation and Problem Solving

- Testing a model on data that was not used during training
- Measuring incorrect predictions
- Interpreting experimental results instead of relying only on the code running successfully
- Understanding that a single accuracy score does not tell the whole story about a model

### Writing Better Projects

I also practiced structuring a Python script, using meaningful variable names, writing English comments and documentation, and preparing a project for publication on GitHub.

## 🚀 How to Run the Project

### Requirements

- Python 3.10 or newer
- pip

### 1. Clone the Repository

Replace `YOUR_REPOSITORY_URL` with the URL of your GitHub repository.

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd machine-learning-projects/iris-flower-classification
```

### 3. Install the Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Program

```bash
python iris_classification.py
```

The program displays the number of training and testing samples, model accuracy, confusion matrix, and number of incorrect predictions.

## 📁 Project Structure

```text
iris-flower-classification/
├── iris_classification.py
├── README.md
└── requirements.txt
```

- `iris_classification.py` — Contains the dataset loading, data splitting, model training, predictions, and evaluation.
- `README.md` — Explains the project, workflow, learning outcomes, and experimental results.
- `requirements.txt` — Lists the required Python dependency.

## 🔭 Future Improvements

This project is a starting point for further experiments. Possible improvements include:

- Testing different values of `n_neighbors`
- Comparing KNN with other classification algorithms
- Using cross-validation to obtain a more reliable performance estimate
- Exploring feature scaling and its effect on KNN
- Visualizing the dataset and classification results
- Allowing users to enter flower measurements and receive a prediction
- Saving the trained model for later use

## 🎯 My Learning Journey

My goal is to strengthen my programming skills by building practical projects and understanding how each part works.

This project helped me move from learning individual Python concepts toward applying them in a complete machine learning workflow: preparing data, training a model, making predictions, and evaluating results.

I plan to build on this foundation by exploring more datasets, learning additional algorithms, and developing more advanced machine learning projects.

## 👨‍💻 Author

**Saman Hooti**

Python Programmer | Aspiring Machine Learning Engineer

*Learning by building, experimenting, and improving one project at a time.*

