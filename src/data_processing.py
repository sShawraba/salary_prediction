print("RUNNING DATA PROCESSING FILE")
import pandas as pd
import os
from sklearn.model_selection import train_test_split

CLEANED_DATA_PATH = "data/cleaned/cleaned_data.csv"
df = pd.read_csv("data/raw/ds_salaries.csv")
df = df.rename(columns={'salary_in_usd': 'target_salary_in_usd'})
df.drop(columns=['Unnamed: 0', 'salary', 'salary_currency', 'employee_residence'], inplace = True)

# experience level - ordinal
exp_map = {
    "EN": 0,  # Entry
    "MI": 1,  # Mid
    "SE": 2,  # Senior
    "EX": 3   # Executive
}
df["experience_level"] = df["experience_level"].map(exp_map)

#df["remote_ratio"] = df["remote_ratio"] / 100

# company size - ordinal
size_map = {
    "S": 0,
    "M": 1,
    "L": 2
}
df["company_size"] = df["company_size"].map(size_map)

# country - choose top 5 so that we can one hot encode
top_countries = df["company_location"].value_counts().nlargest(5).index

for country in top_countries:
    df[f'country_{country}'] = (df['company_location'] == country).astype(int)
df.drop(columns='company_location', inplace=True)

# job title - group so that we can one hot encode
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

df["job_category"] = df["job_title"].apply(simplify_title)
df = df.drop(columns=["job_title"])

# one hot encode the job_category + employement_type
df = pd.get_dummies(df, dtype=int, columns=[
    "employment_type",
    "job_category"
], drop_first=True)

# dplit data
X = df.drop(columns=["target_salary_in_usd"])
y = df["target_salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#save to path
os.makedirs("data/cleaned", exist_ok=True)

train_df = X_train.copy()
train_df["target_salary_in_usd"] = y_train

test_df = X_test.copy()
test_df["target_salary_in_usd"] = y_test

train_df.to_csv("data/cleaned/train.csv", index=False)
test_df.to_csv("data/cleaned/test.csv", index=False)

print("Train and test data saved successfully!")
