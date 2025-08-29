
# src/data_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_titanic(url: str = TITANIC_URL):
    df = pd.read_csv(url)
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived", "Ticket", "Name", "PassengerId", "Cabin"])

    print("--- Vista Previa del DataFrame X Procesado ---")
    print(X.head())  # .head() muestra las primeras 5 filas
    print("\n--- Información y Tipos de Datos de X ---")
    print(X.info())  # .info() te muestra las columnas, si hay nulos y el tipo de dato
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, num_cols, cat_cols

def make_preprocessor(num_cols, cat_cols):
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )
    return pre

def train_test_split_strat(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
