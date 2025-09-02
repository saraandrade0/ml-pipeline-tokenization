import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path
from utils import load_csv, ensure_dir

df = load_csv("data/samples.csv")  # cols: id, texto, label
col_name = "texto" if "texto" in df.columns else "text"
X_text = df[col_name].fillna("")


vectorizer = TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

ensure_dir("models")
joblib.dump(vectorizer, "models/tfidf.joblib")
pd.DataFrame({"n_features":[X.shape[1]]}).to_csv("models/tokenizacao_meta.csv", index=False)
print("Tokenização ok,", X.shape)
