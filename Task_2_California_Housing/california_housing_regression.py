# California Housing: OLS vs Ridge vs Lasso

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# folder where the graphs and tables get saved
os.makedirs("results", exist_ok=True)

# ---------- 1. load the data ----------
df = pd.read_csv("ridge.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())      # only total_bedrooms has missing values (207)
print(df["ocean_proximity"].value_counts())

# ---------- 2. correlation heatmap ----------
num_cols = df.drop(columns=["ocean_proximity"]).columns

plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=150)
plt.show()
# total_rooms, total_bedrooms, population and households are all very
# correlated with each other -> multicollinearity problem

# ---------- 3. checking multicollinearity with VIF ----------
def vif_table(data):
    # VIF = 1 / (1 - R^2) where each feature is predicted from the others
    vifs = {}
    for col in data.columns:
        others = data.drop(columns=col)
        r2 = LinearRegression().fit(others, data[col]).score(others, data[col])
        vifs[col] = 1 / (1 - r2)
    return pd.Series(vifs).sort_values(ascending=False).round(2)

features_only = df.drop(columns=["median_house_value", "ocean_proximity"]).dropna()
print("VIF before feature engineering:")
print(vif_table(features_only))
# VIF above 10 is usually considered a problem

# ---------- 4. feature engineering ----------
# ratios per household are more meaningful than the raw totals
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# interaction features
df["income_x_rooms"] = df["median_income"] * df["rooms_per_household"]
df["income_x_age"] = df["median_income"] * df["housing_median_age"]

# the raw totals were the collinear ones, so drop them
df = df.drop(columns=["total_rooms", "total_bedrooms", "households", "population"])

new_features = df.drop(columns=["median_house_value", "ocean_proximity"]).dropna()
print("\nVIF after feature engineering:")
print(vif_table(new_features))
# the big VIFs from total_bedrooms/households are gone. median_income and
# income_x_rooms are still high because an interaction term always contains
# its parent variable, so some correlation there is expected

# ---------- 5. split, impute, scale, encode ----------
X = df.drop(columns="median_house_value")
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

num_features = X.drop(columns="ocean_proximity").columns.tolist()
cat_features = ["ocean_proximity"]

# median imputing (bedrooms_per_room has the NaNs now) + standardizing
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# one-hot for the categorical column
preprocess = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])
# putting everything in a pipeline so the imputer/scaler are fit on the
# training data only (and inside each CV fold too)

# ---------- 6. OLS ----------
ols = Pipeline([("prep", preprocess), ("model", LinearRegression())])
ols.fit(X_train, y_train)

# ---------- 7. Ridge and Lasso with GridSearchCV ----------
ridge_pipe = Pipeline([("prep", preprocess), ("model", Ridge())])
ridge_grid = GridSearchCV(
    ridge_pipe,
    param_grid={"model__alpha": np.logspace(-3, 3, 13)},
    cv=5, scoring="neg_root_mean_squared_error")
ridge_grid.fit(X_train, y_train)

lasso_pipe = Pipeline([("prep", preprocess), ("model", Lasso(max_iter=100000))])
lasso_grid = GridSearchCV(
    lasso_pipe,
    param_grid={"model__alpha": np.logspace(-1, 3, 9)},
    cv=5, scoring="neg_root_mean_squared_error")
lasso_grid.fit(X_train, y_train)

print("\nBest Ridge alpha:", ridge_grid.best_params_["model__alpha"])
print("Best Lasso alpha:", lasso_grid.best_params_["model__alpha"])

# ---------- 8. compare on the test set ----------
models = {
    "OLS": ols,
    "Ridge": ridge_grid.best_estimator_,
    "Lasso": lasso_grid.best_estimator_,
}

rows = []
preds = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    preds[name] = y_pred
    rows.append([
        np.sqrt(mean_squared_error(y_test, y_pred)),
        mean_absolute_error(y_test, y_pred),
        r2_score(y_test, y_pred),
    ])

results = pd.DataFrame(rows, index=models.keys(), columns=["RMSE", "MAE", "R2"])
print("\nTest set results:")
print(results.round(3))
results.round(3).to_csv("results/model_comparison.csv")

# ---------- 9. what did the coefficients do? ----------
feature_names = (num_features +
                 list(ols.named_steps["prep"].named_transformers_["cat"]
                      .get_feature_names_out(cat_features)))

coefs = pd.DataFrame({
    "OLS": ols.named_steps["model"].coef_,
    "Ridge": ridge_grid.best_estimator_.named_steps["model"].coef_,
    "Lasso": lasso_grid.best_estimator_.named_steps["model"].coef_,
}, index=feature_names)
print("\nCoefficients:")
print(coefs.round(0))
coefs.round(0).to_csv("results/coefficients.csv")
print("\nFeatures Lasso set to zero:", list(coefs.index[coefs["Lasso"] == 0]))

# ---------- 10. predicted vs actual ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
lims = [y_test.min(), y_test.max()]

for ax, (name, y_pred) in zip(axes, preds.items()):
    ax.scatter(y_test, y_pred, alpha=0.3, s=8)
    ax.plot(lims, lims, color="red", linestyle="--")    # perfect prediction line
    ax.set_title(name)
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")

plt.tight_layout()
plt.savefig("results/predicted_vs_actual.png", dpi=150)
plt.show()

# ---------- 11. residual distributions ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, y_pred) in zip(axes, preds.items()):
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, ax=ax, bins=50)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{name} residuals")
    ax.set_xlabel("Residual (actual - predicted)")

plt.tight_layout()
plt.savefig("results/residual_distributions.png", dpi=150)
plt.show()

# ---------- Conclusion ----------
# All three models give almost the same scores (R2 around 0.61). Replacing the
# raw totals with per-household ratios removed the worst multicollinearity
# (VIF of 35 down to under 10 for those columns). The interaction terms bring a
# little collinearity back, but with 20,000 rows and only a few features, Ridge
# and Lasso have very little to fix compared to plain OLS. Lasso only set one
# dummy (NEAR BAY) to zero, and Ridge just shrank the coefficients slightly.
# R2 is moderate because a linear model can't capture the non-linear location
# effect, and median_house_value is capped at 500001, which shows up as a
# vertical column of points at the right edge of the predicted vs actual plot and skews the
# residuals.
