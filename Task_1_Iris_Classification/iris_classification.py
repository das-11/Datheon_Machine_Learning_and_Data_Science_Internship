
# ==========================================================
# TASK 1: IRIS CLASSIFICATION
# Logistic Regression vs Decision Tree
# Datheon Machine Learning & Data Science Internship
# ==========================================================

# 1. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 2. Create images folder

os.makedirs("images", exist_ok=True)


# 3. Load dataset

df = pd.read_csv("iris.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Distribution:")
print(df["species"].value_counts())

print("\nStatistical Summary:")
print(df.describe())


# 4. Exploratory Data Analysis - Pairplot

pair = sns.pairplot(df, hue="species")

pair.figure.suptitle(
    "Iris Dataset - Pairplot",
    y=1.02
)

pair.savefig(
    "images/pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.savefig("pairplot.png", dpi=300, bbox_inches="tight")
plt.close()


# 5. Separate features and target

X = df.drop("species", axis=1)
y = df["species"]


# 6. Train-Test Split (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 7. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# 8. Train Logistic Regression Model

log_reg = LogisticRegression(max_iter=200)

log_reg.fit(X_train_scaled, y_train)

lr_pred = log_reg.predict(X_test_scaled)


# 9. Train Decision Tree Model

tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree.fit(X_train_scaled, y_train)

dt_pred = tree.predict(X_test_scaled)


# 10. Model Evaluation

def evaluate_model(y_true, y_pred):

    return [
        accuracy_score(y_true, y_pred),
        precision_score(
            y_true, y_pred, average="macro"
        ),
        recall_score(
            y_true, y_pred, average="macro"
        ),
        f1_score(
            y_true, y_pred, average="macro"
        )
    ]


results = pd.DataFrame(
    [
        evaluate_model(y_test, lr_pred),
        evaluate_model(y_test, dt_pred)
    ],
    index=[
        "Logistic Regression",
        "Decision Tree"
    ],
    columns=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
)

print("\nModel Performance Comparison:")
print(results.round(3))

results.to_csv("model_comparison.csv")


# 11. Classification Reports

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

print("\nDecision Tree Report:")
print(classification_report(y_test, dt_pred))


# 12. Confusion Matrix Heatmaps

labels = ["setosa", "versicolor", "virginica"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    confusion_matrix(y_test, lr_pred, labels=labels),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[0]
)

axes[0].set_title("Logistic Regression")

sns.heatmap(
    confusion_matrix(y_test, dt_pred, labels=labels),
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[1]
)

axes[1].set_title("Decision Tree")

for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# 13. Decision Boundary Visualization

# Use petal length and petal width

features = ["petal_length", "petal_width"]

X_train_2 = X_train[features]
X_test_2 = X_test[features]

# Scale selected features

scaler_2 = StandardScaler()

X_train_2_scaled = scaler_2.fit_transform(X_train_2)

X_test_2_scaled = scaler_2.transform(X_test_2)

# Convert species names into numerical labels

species_mapping = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}

y_train_num = y_train.map(species_mapping)
y_test_num = y_test.map(species_mapping)

# Train models using two features

lr_2 = LogisticRegression(max_iter=200)

lr_2.fit(X_train_2_scaled, y_train_num)

dt_2 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

dt_2.fit(X_train_2_scaled, y_train_num)


# Plot decision boundaries

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = [
    (axes[0], lr_2, "Logistic Regression"),
    (axes[1], dt_2, "Decision Tree")
]

for ax, model, title in models:

    DecisionBoundaryDisplay.from_estimator(
        model,
        X_train_2_scaled,
        cmap="Set2",
        alpha=0.5,
        ax=ax
    )

    ax.scatter(
        X_test_2_scaled[:, 0],
        X_test_2_scaled[:, 1],
        c=y_test_num,
        cmap="Set2",
        edgecolor="black"
    )

    ax.set_title(title)

    ax.set_xlabel("Petal Length (Scaled)")
    ax.set_ylabel("Petal Width (Scaled)")

plt.tight_layout()

plt.savefig(
    "images/decision_boundaries.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.savefig("decision_boundaries.png", dpi=300, bbox_inches="tight")
plt.close()


# 14. Final Conclusion

print("\n========== FINAL CONCLUSION ==========")

print(results.round(3))

print(
    "\nBoth models were trained and evaluated "
    "using the Iris dataset."
)

print(
    "Performance was compared using Accuracy, "
   "Precision")



