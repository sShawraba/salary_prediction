import pandas as pd
import pickle
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
train_df = pd.read_csv("data/cleaned/train.csv")
test_df = pd.read_csv("data/cleaned/test.csv")

X_train = train_df.drop(columns=["target_salary_in_usd"])
y_train = np.log1p(train_df["target_salary_in_usd"])


X_test = test_df.drop(columns=["target_salary_in_usd"])
y_test = np.log1p(test_df["target_salary_in_usd"])

# Model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("RMSE:", rmse)
print("R2:", r2)

#Save model
os.makedirs("src/models/saved", exist_ok=True)
with open("src/models/saved/decision_tree.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

#save the columns order of features
with open("src/models/saved/features.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Feature columns saved!")
