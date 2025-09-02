# score.py
import sys
import numpy as np
import pandas as pd
import joblib

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)  # remove BOM
        .str.strip()
        .str.lower()
    )
    return df

def resolve_text_col(df: pd.DataFrame) -> str:
    text_candidates = ["texto", "text", "mensagem", "descricao", "descrição", "message", "content", "body"]
    for c in text_candidates:
        if c in df.columns:
            return c
    # fallback: first object-like column that isn't obviously id/label
    fallback = [c for c in df.columns if df[c].dtype == object and c not in {"id", "label", "classe", "target", "y"}]
    if fallback:
        return fallback[0]
    raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

def resolve_id_col(df: pd.DataFrame) -> str:
    id_candidates = ["id", "doc_id", "row_id", "uid"]
    for c in id_candidates:
        if c in df.columns:
            return c
    # create a synthetic id if none exists
    df["id"] = np.arange(len(df), dtype=int)
    return "id"

def to_probabilities(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        # binary: take positive class (assumed index 1)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
    # fallback for linear models without predict_proba
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        # sigmoid map to (0,1)
        return 1.0 / (1.0 + np.exp(-z))
    # last resort
    if hasattr(clf, "predict"):
        yhat = clf.predict(X)
        return yhat.astype(float)
    raise ValueError("Classifier does not support scoring via predict_proba/decision_function/predict.")

def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "data/samples.csv"
    df = pd.read_csv(infile)
    df = normalize_columns(df)

    text_col = resolve_text_col(df)
    id_col = resolve_id_col(df)

    vec = joblib.load("models/tfidf.joblib")
    clf = joblib.load("models/modelo_lr.joblib")

    X = vec.transform(df[text_col].fillna(""))

    scores = to_probabilities(clf, X)
    df_out = pd.DataFrame({ "id": df[id_col], "score": scores })

    print(df_out.head(10))
    df_out.to_csv("scores.csv", index=False)
    print("Scoring ok. Wrote scores.csv with", len(df_out), "rows.")

if __name__ == "__main__":
    main()
