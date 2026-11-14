# Iris classification: Logistic Regression vs Decision Tree

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ---------- 1. load the data ----------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.isnull().sum())          # checking for missing values
print(df["species"].value_counts())
print(df.describe())

# ---------- 2. pairplot ----------
sns.pairplot(df, hue="species")
plt.show()
# setosa is clearly separated, versicolor and virginica overlap a bit

# ---------- 3. split into X and y, then train/test (80/20) ----------
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 4. scaling ----------
# fit only on the training data, otherwise info from the test set leaks in
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ---------- 5. train both models ----------
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_sc, y_train)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train_sc, y_train)

lr_pred = log_reg.predict(X_test_sc)
dt_pred = tree.predict(X_test_sc)

# ---------- 6. compare the metrics ----------
def get_scores(y_true, y_pred):
    # average="macro" since there are 3 classes
    return [accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro"),
            recall_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average="macro")]

results = pd.DataFrame(
    [get_scores(y_test, lr_pred), get_scores(y_test, dt_pred)],
    index=["Logistic Regression", "Decision Tree"],
    columns=["Accuracy", "Precision", "Recall", "F1-score"])
print(results.round(3))

print("\nLogistic Regression report")
print(classification_report(y_test, lr_pred))
print("Decision Tree report")
print(classification_report(y_test, dt_pred))

# ---------- 7. confusion matrices ----------
labels = ["setosa", "versicolor", "virginica"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(confusion_matrix(y_test, lr_pred, labels=labels), annot=True,
            fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
            ax=axes[0])
axes[0].set_title("Logistic Regression")

sns.heatmap(confusion_matrix(y_test, dt_pred, labels=labels), annot=True,
            fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
            ax=axes[1])
axes[1].set_title("Decision Tree")

for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ---------- 8. decision boundaries ----------
# can only plot 2 features, so I use petal length and petal width here
# (they separate the species best). The models are retrained on just these two.
cols = ["petal_length", "petal_width"]
sc2 = StandardScaler()
X2_train = sc2.fit_transform(X_train[cols])
X2_test = sc2.transform(X_test[cols])

# the plot needs numbers for the classes, not names
y_train_num = y_train.map({"setosa": 0, "versicolor": 1, "virginica": 2})
y_test_num = y_test.map({"setosa": 0, "versicolor": 1, "virginica": 2})

lr2 = LogisticRegression(max_iter=200).fit(X2_train, y_train_num)
dt2 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X2_train, y_train_num)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, model, title in [(axes[0], lr2, "Logistic Regression"),
                         (axes[1], dt2, "Decision Tree")]:
    DecisionBoundaryDisplay.from_estimator(model, X2_train, cmap="Set2",
                                           alpha=0.5, ax=ax)
    ax.scatter(X2_test[:, 0], X2_test[:, 1], c=y_test_num, cmap="Set2",
               edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Petal length (scaled)")
    ax.set_ylabel("Petal width (scaled)")

plt.tight_layout()
plt.show()

# ---------- Conclusion ----------
# Both models do well. Setosa is classified perfectly, and almost all the
# mistakes are between versicolor and virginica. The test set only has 30 rows,
# so one wrong prediction changes accuracy by about 3%. The difference between
# the two models is too small to say one is really better.
