
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import load_csv

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase, strip spaces, and remove BOM if present
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df

def resolve_text_and_label_columns(df: pd.DataFrame) -> tuple[str, str]:
    # Accept common synonyms for text/label
    text_candidates = ["texto", "text", "mensagem", "descricao", "descrição", "message", "content", "body"]
    label_candidates = ["label", "classe", "target", "y"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        # Fallback: pick first object/string-like column that isn't obviously an id/label
        fallback = [c for c in df.columns if df[c].dtype == object and c not in {"id", "label", "classe", "target", "y"}]
        if fallback:
            text_col = fallback[0]
        else:
            raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column found. Available columns: {list(df.columns)}")

    return text_col, label_col

# ----- Load data
df = load_csv("data/samples.csv")
df = normalize_columns(df)
text_col, label_col = resolve_text_and_label_columns(df)

X_text = df[text_col].fillna("")
# Be strict: convert labels to int
y = df[label_col].astype(int)

# ----- Load vectorizer (must match the one trained in tokenization.py)
vectorizer = joblib.load("models/tfidf.joblib")
X = vectorizer.transform(X_text)

# ----- Train/test split (guard for tiny datasets / single-class issues)
if len(df[label_col].unique()) < 2:
    raise ValueError("Training requires at least two classes in the label column.")

test_size = 0.2 if len(df) >= 10 else 0.5
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# ----- Train model
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, ytr)

# ----- Save and evaluate
joblib.dump(clf, "models/modelo_lr.joblib")

pred = clf.predict(Xte)
print(classification_report(yte, pred))
print("Training ok. Shapes:", Xtr.shape, Xte.shape)
