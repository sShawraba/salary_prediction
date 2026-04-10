from fastapi import APIRouter
import pickle
import pandas as pd
import numpy as np

router = APIRouter()

# Load model
with open("src/models/saved/decision_tree.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("src/models/saved/features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Job category mapping (same as preprocessing)
def simplify_title(title):
    title = title.lower()
    if "data scientist" in title:
        return "data_scientist"
    elif "engineer" in title:
        return "data_engineer"
    elif "analyst" in title:
        return "data_analyst"
    elif "machine learning" in title or "ml" in title:
        return "ml_engineer"
    elif "manager" in title or "lead" in title:
        return "manager"
    else:
        return "other"

# Route for prediction
@router.get("/predict")
def predict(
    experience_level: int,
    company_size: int,
    remote_ratio: int,
    job_title: str,
    employment_type: str,
    company_location: str
):
    # Build a single-row dataframe
    df = pd.DataFrame({
        "experience_level": [experience_level],
        "company_size": [company_size],
        "remote_ratio": [remote_ratio],
        "employment_type": [employment_type],
        "job_category": [simplify_title(job_title)],
        "company_location": [company_location]
    })

    # One-hot encode job_category and employment_type
    df = pd.get_dummies(df, columns=["job_category", "employment_type"], drop_first=True)

    # One-hot encode top 5 countries (like in training)
    top_countries = [c for c in feature_columns if c.startswith("country_")]
    for country_col in top_countries:
        country_name = country_col.replace("country_", "")
        df[country_col] = int(df.get("company_location")[0] == country_name)
    df.drop(columns=["company_location"], inplace=True)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order
    df = df[feature_columns]

    # Predict
    y_pred = model.predict(df)
    salary_pred = np.expm1(y_pred)[0]  # reverse log transform

    return {"predicted_salary_usd": round(salary_pred, 2)}