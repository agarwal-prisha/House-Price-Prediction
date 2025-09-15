House Price Prediction using ML

Overview
This project predicts house prices using the Boston Housing Dataset.  
It compares multiple ML models:Linear Regression, Decision Tree, Random Forest, and evaluates their performance.

Dataset
Dataset Source:Kaggle Boston Housing Dataset - https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset
Target variable:`medv` (Median value of owner-occupied homes in $1000s)

Preprocessing Steps
Explanation:
Selected features (X) and target (y = medv).
Split dataset into train (80%) and test (20%) sets.
Handled missing values using mean imputation.
Scaled numerical features for Linear Regression (important for gradient-based models).
No scaling required for tree-based models (Decision Tree, Random Forest).

Feature and target separation:
X = data.drop("medv", axis=1)
y = data["medv"]

Train-test split:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Handling missing values:
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

Feature scaling:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Models Used
Linear Regression
Decision Tree Regressor
Random Forest Regressor

Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt


