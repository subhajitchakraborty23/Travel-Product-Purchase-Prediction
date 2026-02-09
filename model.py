import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# ======================
# Load data
# ======================
df = pd.read_csv("Travel.csv")

df.drop(['CustomerID'], axis=1, inplace=True)

# ======================
# Missing value handling
# ======================
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['TypeofContact'].fillna(df['TypeofContact'].mode()[0], inplace=True)
df['DurationOfPitch'].fillna(df['DurationOfPitch'].mean(), inplace=True)
df['Occupation'].fillna(df['Occupation'].mode()[0], inplace=True)
df['NumberOfFollowups'].fillna(df['NumberOfFollowups'].mean(), inplace=True)
df['PreferredPropertyStar'].fillna(df['PreferredPropertyStar'].mean(), inplace=True)
df['MaritalStatus'].fillna(df['MaritalStatus'].mode()[0], inplace=True)
df['NumberOfTrips'].fillna(df['NumberOfTrips'].mean(), inplace=True)
df['NumberOfChildrenVisiting'].fillna(df['NumberOfChildrenVisiting'].mean(), inplace=True)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean(), inplace=True)

# ======================
# Cleaning
# ======================
df["Gender"] = df["Gender"].replace({'Fe Male': 'Female'})
df["MaritalStatus"] = df["MaritalStatus"].replace({'Unmarried': 'Single'})

# ======================
# Feature Engineering
# ======================
df['TotalVisits'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

# ======================
# Split
# ======================
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Preprocessor
# ======================
cat_features = X.select_dtypes(include='object').columns
num_features = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ]
)

# ======================
# Final Model
# ======================
rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

# ======================
# Pipeline (THIS IS THE KEY)
# ======================
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ]
)

# ======================
# Train
# ======================
pipeline.fit(X_train, y_train)

# ======================
# Evaluation
# ======================
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test Recall:", recall_score(y_test, y_test_pred))
print("Test F1:", f1_score(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_test_pred))

# ======================
# Save model (CORRECT WAY)
# ======================
joblib.dump(pipeline, "model.pkl")

print("✅ Model saved as model.pkl")
