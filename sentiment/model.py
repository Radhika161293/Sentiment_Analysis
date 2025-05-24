
import pandas as pd
import numpy as np
import joblib
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# --- Load Data ---
df = pd.read_csv("chatgpt_reviews_cleaned.csv")
df["review"] = df["review"].astype(str)

# --- Clean & Preprocess Text ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["cleaned_review"] = df["review"].apply(clean_text)

# --- Encode Sentiment Labels ---
le = LabelEncoder()
df["sentiment_label"] = le.fit_transform(df["sentiment"])  # Positive = 2, Neutral = 1, Negative = 0

# --- Split Data ---
X = df["cleaned_review"]
y = df["sentiment_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- TF-IDF Vectorizer ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

# --- Models ---
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=150),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

best_model = None
best_score = 0

print("\n🧠 Model Training and Evaluation:")
for name, clf in models.items():
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    if acc > best_score:
        best_score = acc
        best_model = pipeline

# --- Save Best Model and Vectorizer ---
joblib.dump(best_model.named_steps["clf"], "sentiment_model.pkl")
joblib.dump(best_model.named_steps["tfidf"], "vectorizer.pkl")
print(f"\n✅ Best model saved with accuracy: {best_score:.4f}")
