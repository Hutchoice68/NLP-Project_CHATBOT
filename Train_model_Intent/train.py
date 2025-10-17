# ============================================
# Thai Intent Classification (Improved Version)
# ============================================

import pandas as pd
import numpy as np
import re
import emoji
import seaborn as sns
import matplotlib.pyplot as plt
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv('Data/intents_dataset.csv')
print(f"✅ Loaded dataset: {df.shape}")

# -------------------------------
# 2. Text Preprocessing (Thai)
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # ลดตัวซ้ำ
    text = re.sub(r"[^ก-๙a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_with_stopwords(text, remove_stopwords=False):
    tokens = word_tokenize(text, engine='newmm')
    if remove_stopwords:
        stop_words = set(thai_stopwords())
        tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
df['tokens'] = df['clean_text'].apply(lambda x: tokenize_with_stopwords(x, remove_stopwords=False))

# -------------------------------
# 3. Split Train/Test
# -------------------------------
X = df['tokens']
y = df['intent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# -------------------------------
# 4. Sentence Embedding (Thai model)
# -------------------------------
print("🔧 Loading Thai sentence embedding model (LaBSE)...")
sem_model = SentenceTransformer('sentence-transformers/LaBSE')

emb_train = sem_model.encode(X_train.tolist(), show_progress_bar=True)
emb_test = sem_model.encode(X_test.tolist(), show_progress_bar=True)
print("✅ Semantic embeddings created.")

# -------------------------------
# 5. TF-IDF (Lexical)
# -------------------------------
tfidf = TfidfVectorizer(max_features=800)
tfidf_train = tfidf.fit_transform(X_train).toarray()
tfidf_test = tfidf.transform(X_test).toarray()
print("✅ TF-IDF features created.")

# -------------------------------
# 6. Feature Fusion (Weighted + Normalized)
# -------------------------------
scaler_sem = StandardScaler()
scaler_tfidf = StandardScaler()

emb_train_scaled = scaler_sem.fit_transform(emb_train)
emb_test_scaled = scaler_sem.transform(emb_test)

tfidf_train_scaled = scaler_tfidf.fit_transform(tfidf_train)
tfidf_test_scaled = scaler_tfidf.transform(tfidf_test)

alpha, beta = 0.7, 0.3  # semantic:lexical weighting

train_features = np.concatenate([
    emb_train_scaled * alpha,
    tfidf_train_scaled * beta
], axis=1)

test_features = np.concatenate([
    emb_test_scaled * alpha,
    tfidf_test_scaled * beta
], axis=1)

print("✅ Weighted feature fusion complete:", train_features.shape)

# -------------------------------
# 7. Handle Imbalanced Data
# -------------------------------
print("🔍 Intent distribution before balancing:")
print(Counter(y_train))

smote = SMOTE(random_state=42)
train_features_balanced, y_train_balanced = smote.fit_resample(train_features, y_train)

print("✅ After SMOTE balancing:")
print(Counter(y_train_balanced))

# -------------------------------
# 8. Train Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', C=0.5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        class_weight='balanced', random_state=42
    ),
    "SVM": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True
    )
}

ensemble = VotingClassifier(
    estimators=[
        ('lr', models["Logistic Regression"]),
        ('rf', models["Random Forest"]),
    ],
    voting='soft'
)
models["Ensemble"] = ensemble

results = {}
for name, model in models.items():
    print(f"\n🚀 Training {name} ...")
    model.fit(train_features_balanced, y_train_balanced)
    y_pred = model.predict(test_features)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "acc": acc, "y_pred": y_pred}
    print(f"🎯 Accuracy: {acc:.4f}")

# -------------------------------
# 9. Evaluate Best Model
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]['acc'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['acc']:.4f})")
print(classification_report(y_test, y_pred_best))

# -------------------------------
# 10. Per-class Evaluation
# -------------------------------
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_best, average=None, labels=sorted(y.unique())
)
perf_df = pd.DataFrame({
    'Intent': sorted(y.unique()),
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
print("\n📊 Per-Intent Performance:")
print(perf_df.sort_values('F1-Score'))

problematic = perf_df[perf_df['F1-Score'] < 0.7]
if len(problematic) > 0:
    print("\n⚠️ Intents needing improvement:")
    print(problematic)

# -------------------------------
# 11. Confusion Matrix
# -------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_test, y_pred_best),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique())
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 12. Misclassified Examples
# -------------------------------
mis_idx = np.where(y_test != y_pred_best)[0]
print(f"\n❌ Misclassified examples ({len(mis_idx)}):")
for i in mis_idx[:5]:
    print(f"Text: {X_test.iloc[i]}")
    print(f"True: {y_test.iloc[i]} | Pred: {y_pred_best[i]}")
    print("---")

# -------------------------------
# 13. Save Model Bundle
# -------------------------------
bundle = {
    "semantic": sem_model,
    "tfidf": tfidf,
    "clf": best_model,
    "scaler_sem": scaler_sem,
    "scaler_tfidf": scaler_tfidf,
    "alpha": alpha,
    "beta": beta
}

with open("models/thai_intent_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n💾 Saved best model successfully.")
