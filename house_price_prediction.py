
# House Price Prediction ACM task

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Loading Dataset
data = pd.read_csv("boston.csv")
print("First 5 rows of dataset:")
print(data.head())

#Basic Data Exploration
print("\nDataset Info:")
print(data.info())

print("\nMissing values in dataset:")
print(data.isnull().sum())

print("\nDataset statistics:")
print(data.describe())

#simple visualization
sns.histplot(data["medv"], bins=30, kde=True)
plt.title("Distribution of House Prices (medv)")
plt.savefig("results/house_price_distribution.png")
plt.show()

#Preprocessing

#Features(X) and target(y)
X = data.drop("medv", axis=1)
y = data["medv"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#Scaling features (important for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100)
}

results = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)   #needs scaled data
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)          #tree models don’t need scaling
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

#Evaluation Results
print("\nModel Evaluation Results:")
for model, metrics in results.items():
    print(f"{model}: MSE={metrics['MSE']:.2f}, R2={metrics['R2']:.2f}")

#Visualizatiing the Results(additional)
result_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(8,5))   
result_df.plot(kind="bar", ax=ax)       
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score (Lower MSE is better, Higher R2 is better)")
ax.set_xticklabels(result_df.index,rotation=0,ha='center')
fig.savefig("results/model_performance.png")   # save using fig
plt.show()

with open("results/evaluation.txt", "w") as f:
    for model, metrics in results.items():
        f.write(f"{model}: MSE={metrics['MSE']:.2f}, R2={metrics['R2']:.2f}\n")

