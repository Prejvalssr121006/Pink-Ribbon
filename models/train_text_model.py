import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("/Users/adityachaubey/Downloads/data.csv")

# =========================
# CLEAN DATA
# =========================

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert all columns to numeric if possible
df = df.apply(pd.to_numeric, errors='ignore')

# =========================
# TARGET COLUMN FIX
# =========================
# Make sure diagnosis exists
if "diagnosis" not in df.columns:
    raise Exception("Column 'diagnosis' not found in dataset")

# Convert labels
df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})

# =========================
# SPLIT FEATURES & LABEL
# =========================
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# =========================
# HANDLE MISSING VALUES (IMPORTANT FIX)
# =========================
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
with open("text_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Text model trained successfully and saved as text_model.pkl")