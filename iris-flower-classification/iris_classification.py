
"""Classify Iris flowers using the K-Nearest Neighbors algorithm."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create and train the KNN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Predict the species of the test samples
    predictions = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    errors = (y_test != predictions).sum()

    # Display the results
    print("=== Iris Flower Classification ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Model accuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    print(matrix)

    print(f"\nIncorrect predictions: {errors}")


if __name__ == "__main__":
    main()
