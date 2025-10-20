# ============================================
# Thai Intent Classification with Complete Analysis
# ============================================

import pandas as pd
import numpy as np
import re
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
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')

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
    text = re.sub(r'(.)\1{2,}', r'\1', text)
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
# 3. Data Analysis & Split
# -------------------------------
X = df['tokens']
y = df['intent']

# Check dataset size and class distribution
print("\n📊 Dataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Number of intents: {y.nunique()}")
print(f"\nClass distribution:")
print(y.value_counts().sort_index())

# Check for duplicates
duplicates = df.duplicated(subset=['text']).sum()
print(f"\n⚠️ Duplicate texts: {duplicates}")
if duplicates > 0:
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['text'])
    X = df['tokens']
    y = df['intent']

# Increase test size for better generalization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# -------------------------------
# 4. Create Features
# -------------------------------
print("\n🔧 Creating Features...")

# 4.1 Semantic Embeddings (LaBSE)
print("Loading Thai sentence embedding model (LaBSE)...")
sem_model = SentenceTransformer('sentence-transformers/LaBSE')
emb_train = sem_model.encode(X_train.tolist(), show_progress_bar=True)
emb_test = sem_model.encode(X_test.tolist(), show_progress_bar=True)
print("✅ Semantic embeddings created.")

# 4.2 TF-IDF Features
tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
tfidf_train = tfidf.fit_transform(X_train).toarray()
tfidf_test = tfidf.transform(X_test).toarray()
print("✅ TF-IDF features created.")

# Check feature dimensions
print(f"   TF-IDF shape: {tfidf_train.shape}")
print(f"   Semantic shape: {emb_train.shape}")

# -------------------------------
# 5. Handle Imbalanced Data
# -------------------------------
print("\n🔍 Intent distribution:")
print(Counter(y_train))

# Check if SMOTE is needed
min_samples = min(Counter(y_train).values())
print(f"\n⚠️ Minimum samples per class: {min_samples}")

if min_samples >= 6:
    smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
    use_smote = True
else:
    print("⚠️ SMOTE disabled - insufficient samples for some classes")
    use_smote = False

# -------------------------------
# 6. Prepare Model Configurations
# -------------------------------
models_dict = {
    "Logistic Regression": LogisticRegression(
        max_iter=500, 
        class_weight='balanced', 
        C=0.1,  # Stronger regularization
        solver='saga',
        penalty='l2'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,  # Reduced from 300
        max_depth=10,      # Reduced from 20
        min_samples_split=10,  # Increased from 5
        min_samples_leaf=5,    # Added constraint
        class_weight='balanced', 
        random_state=42
    ),
    "SVM": SVC(
        kernel='rbf', 
        C=0.5,  # Reduced from 1.0
        gamma='scale', 
        class_weight='balanced', 
        probability=True
    )
}

# -------------------------------
# 7. BASELINE 1: TF-IDF Only
# -------------------------------
print("\n" + "="*60)
print("📊 4.2 BASELINE 1: TF-IDF Only")
print("="*60)

scaler_tfidf_base = StandardScaler()
tfidf_train_scaled = scaler_tfidf_base.fit_transform(tfidf_train)
tfidf_test_scaled = scaler_tfidf_base.transform(tfidf_test)

if use_smote:
    tfidf_train_balanced, y_train_tfidf_balanced = smote.fit_resample(tfidf_train_scaled, y_train)
else:
    tfidf_train_balanced, y_train_tfidf_balanced = tfidf_train_scaled, y_train

tfidf_results = {}
for name, model in models_dict.items():
    print(f"\n🚀 Training {name} with TF-IDF...")
    model_clone = type(model)(**model.get_params())
    model_clone.fit(tfidf_train_balanced, y_train_tfidf_balanced)
    y_pred = model_clone.predict(tfidf_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    tfidf_results[name] = {
        'model': model_clone, 'accuracy': acc, 'precision': p, 
        'recall': r, 'f1_score': f1, 'y_pred': y_pred
    }
    print(f"   Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# -------------------------------
# 8. BASELINE 2: S-BERT Only
# -------------------------------
print("\n" + "="*60)
print("📊 4.2 BASELINE 2: S-BERT (LaBSE) Only")
print("="*60)

scaler_sem_base = StandardScaler()
emb_train_scaled = scaler_sem_base.fit_transform(emb_train)
emb_test_scaled = scaler_sem_base.transform(emb_test)

if use_smote:
    emb_train_balanced, y_train_emb_balanced = smote.fit_resample(emb_train_scaled, y_train)
else:
    emb_train_balanced, y_train_emb_balanced = emb_train_scaled, y_train

sbert_results = {}
for name, model in models_dict.items():
    print(f"\n🚀 Training {name} with S-BERT...")
    model_clone = type(model)(**model.get_params())
    model_clone.fit(emb_train_balanced, y_train_emb_balanced)
    y_pred = model_clone.predict(emb_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    sbert_results[name] = {
        'model': model_clone, 'accuracy': acc, 'precision': p,
        'recall': r, 'f1_score': f1, 'y_pred': y_pred
    }
    print(f"   Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# -------------------------------
# 9. HYBRID FUSION
# -------------------------------
print("\n" + "="*60)
print("📊 4.3 HYBRID EMBEDDING FUSION")
print("="*60)

# Test different fusion weights
fusion_weights = [
    (0.5, 0.5, "Equal"),
    (0.6, 0.4, "Semantic-Heavy"),
    (0.7, 0.3, "Semantic-Dominant"),
    (0.4, 0.6, "Lexical-Heavy")
]

fusion_results = {}

for alpha, beta, weight_name in fusion_weights:
    print(f"\n🔬 Testing Fusion: {weight_name} (α={alpha}, β={beta})")
    
    train_fusion = np.concatenate([
        emb_train_scaled * alpha,
        tfidf_train_scaled * beta
    ], axis=1)
    
    test_fusion = np.concatenate([
        emb_test_scaled * alpha,
        tfidf_test_scaled * beta
    ], axis=1)
    
    if use_smote:
        train_fusion_balanced, y_train_fusion_balanced = smote.fit_resample(train_fusion, y_train)
    else:
        train_fusion_balanced, y_train_fusion_balanced = train_fusion, y_train
    
    fusion_results[weight_name] = {}
    
    for name, model in models_dict.items():
        model_clone = type(model)(**model.get_params())
        model_clone.fit(train_fusion_balanced, y_train_fusion_balanced)
        y_pred = model_clone.predict(test_fusion)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        fusion_results[weight_name][name] = {
            'model': model_clone, 'accuracy': acc, 'precision': p,
            'recall': r, 'f1_score': f1, 'y_pred': y_pred,
            'alpha': alpha, 'beta': beta
        }
        print(f"   {name}: Acc={acc:.4f}, F1={f1:.4f}")

# -------------------------------
# 10. COMPARATIVE ANALYSIS
# -------------------------------
print("\n" + "="*60)
print("📊 COMPARATIVE ANALYSIS TABLE")
print("="*60)

# ⚠️ OVERFITTING WARNING CHECK
print("\n🔍 Checking for overfitting indicators...")
overfitting_warnings = []

for model_name in models_dict.keys():
    for result_dict, name in [(tfidf_results, 'TF-IDF'), (sbert_results, 'S-BERT')]:
        if result_dict[model_name]['accuracy'] >= 0.99:
            overfitting_warnings.append(f"{model_name} ({name}): {result_dict[model_name]['accuracy']:.4f}")
    
    for weight_name in fusion_results.keys():
        if fusion_results[weight_name][model_name]['accuracy'] >= 0.99:
            overfitting_warnings.append(f"{model_name} (Fusion-{weight_name}): {fusion_results[weight_name][model_name]['accuracy']:.4f}")

if len(overfitting_warnings) > 0:
    print("\n⚠️ POTENTIAL OVERFITTING DETECTED:")
    print("The following models achieved suspiciously high accuracy (≥99%):")
    for warning in overfitting_warnings:
        print(f"   • {warning}")
    print("\n💡 Recommendations:")
    print("   1. Check for data leakage (duplicate texts in train/test)")
    print("   2. Increase test set size (currently using 30%)")
    print("   3. Add more regularization to models")
    print("   4. Collect more diverse data")
    print("   5. Use cross-validation for more reliable estimates")
else:
    print("✅ No obvious overfitting detected")

comparison_data = []

for model_name in models_dict.keys():
    # TF-IDF
    comparison_data.append({
        'Model': model_name,
        'Feature Type': 'TF-IDF Only',
        'Accuracy': tfidf_results[model_name]['accuracy'],
        'Precision': tfidf_results[model_name]['precision'],
        'Recall': tfidf_results[model_name]['recall'],
        'F1-Score': tfidf_results[model_name]['f1_score']
    })
    
    # S-BERT
    comparison_data.append({
        'Model': model_name,
        'Feature Type': 'S-BERT Only',
        'Accuracy': sbert_results[model_name]['accuracy'],
        'Precision': sbert_results[model_name]['precision'],
        'Recall': sbert_results[model_name]['recall'],
        'F1-Score': sbert_results[model_name]['f1_score']
    })
    
    # Fusion variants
    for weight_name in fusion_results.keys():
        comparison_data.append({
            'Model': model_name,
            'Feature Type': f'Fusion ({weight_name})',
            'Accuracy': fusion_results[weight_name][model_name]['accuracy'],
            'Precision': fusion_results[weight_name][model_name]['precision'],
            'Recall': fusion_results[weight_name][model_name]['recall'],
            'F1-Score': fusion_results[weight_name][model_name]['f1_score']
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('models/performance_comparison.csv', index=False)
print("\n💾 Saved comparison table to 'models/performance_comparison.csv'")

# Additional statistics
print("\n📈 Performance Statistics:")
print(f"Mean Accuracy across all models: {comparison_df['Accuracy'].mean():.4f}")
print(f"Std Accuracy: {comparison_df['Accuracy'].std():.4f}")
print(f"Mean F1-Score: {comparison_df['F1-Score'].mean():.4f}")
print(f"Best Accuracy: {comparison_df['Accuracy'].max():.4f}")
print(f"Worst Accuracy: {comparison_df['Accuracy'].min():.4f}")

# Check if all models perform similarly (another overfitting sign)
if comparison_df['Accuracy'].std() < 0.01:
    print("\n⚠️ WARNING: Very low variance in accuracy across models!")
    print("   This might indicate:")
    print("   • Dataset is too easy/simple")
    print("   • Possible data leakage")
    print("   • Need more challenging test cases")

# -------------------------------
# 11. VISUALIZATION: Performance Comparison
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Comparison: Single vs Hybrid Embeddings', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    pivot = comparison_df.pivot(index='Model', columns='Feature Type', values=metric)
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    ax.legend(title='Feature Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig('models/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 12. FIND BEST MODEL
# -------------------------------
best_overall = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
print("\n" + "="*60)
print("🏆 BEST OVERALL MODEL")
print("="*60)
print(f"Model: {best_overall['Model']}")
print(f"Feature Type: {best_overall['Feature Type']}")
print(f"Accuracy: {best_overall['Accuracy']:.4f}")
print(f"F1-Score: {best_overall['F1-Score']:.4f}")

# Extract best model
best_model_name = best_overall['Model']
best_feature_type = best_overall['Feature Type']

if 'Fusion' in best_feature_type:
    weight_name = best_feature_type.split('(')[1].split(')')[0]
    best_model = fusion_results[weight_name][best_model_name]['model']
    y_pred_best = fusion_results[weight_name][best_model_name]['y_pred']
    best_alpha = fusion_results[weight_name][best_model_name]['alpha']
    best_beta = fusion_results[weight_name][best_model_name]['beta']
elif 'TF-IDF' in best_feature_type:
    best_model = tfidf_results[best_model_name]['model']
    y_pred_best = tfidf_results[best_model_name]['y_pred']
else:
    best_model = sbert_results[best_model_name]['model']
    y_pred_best = sbert_results[best_model_name]['y_pred']

# -------------------------------
# 13. CONFUSION MATRIX ANALYSIS
# -------------------------------
print("\n" + "="*60)
print("📊 4.4 IN-DEPTH ANALYSIS: Confusion Matrix")
print("="*60)

cm = confusion_matrix(y_test, y_pred_best)
intent_labels = sorted(y.unique())

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=intent_labels, yticklabels=intent_labels,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name} ({best_feature_type})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Intent', fontweight='bold')
plt.ylabel('True Intent', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify most confused pairs
print("\n❌ Most Confused Intent Pairs:")
confused_pairs = []
for i in range(len(intent_labels)):
    for j in range(len(intent_labels)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append({
                'True': intent_labels[i],
                'Predicted': intent_labels[j],
                'Count': cm[i, j]
            })

if len(confused_pairs) > 0:
    confused_df = pd.DataFrame(confused_pairs).sort_values('Count', ascending=False).head(10)
    print(confused_df.to_string(index=False))
else:
    print("✅ No confusion detected - all predictions are correct!")

# -------------------------------
# 14. PER-INTENT PERFORMANCE
# -------------------------------
print("\n" + "="*60)
print("📊 Per-Intent Performance Analysis")
print("="*60)

p, r, f1, support = precision_recall_fscore_support(
    y_test, y_pred_best, labels=intent_labels, average=None
)

perf_df = pd.DataFrame({
    'Intent': intent_labels,
    'Precision': p,
    'Recall': r,
    'F1-Score': f1,
    'Support': support
})
perf_df = perf_df.sort_values('F1-Score')
print(perf_df.to_string(index=False))

problematic = perf_df[perf_df['F1-Score'] < 0.7]
if len(problematic) > 0:
    print("\n⚠️ Intents needing improvement (F1 < 0.7):")
    print(problematic.to_string(index=False))

# -------------------------------
# 15. SEMANTIC COHERENCE ANALYSIS
# -------------------------------
print("\n" + "="*60)
print("📊 4.4 SEMANTIC COHERENCE ANALYSIS")
print("="*60)

# Sample some examples per intent for analysis
coherence_results = []

for intent in intent_labels[:5]:  # Analyze first 5 intents
    intent_samples = X_test[y_test == intent].head(10)
    if len(intent_samples) < 2:
        continue
    
    # Get embeddings
    intent_emb = sem_model.encode(intent_samples.tolist())
    
    # Calculate pairwise cosine similarity
    sim_matrix = cosine_similarity(intent_emb)
    avg_sim = (sim_matrix.sum() - len(intent_samples)) / (len(intent_samples) * (len(intent_samples) - 1))
    
    coherence_results.append({
        'Intent': intent,
        'Avg Intra-Intent Similarity': avg_sim,
        'Samples': len(intent_samples)
    })

coherence_df = pd.DataFrame(coherence_results)
print("\n🔍 Intra-Intent Semantic Coherence (Higher = More Coherent):")
print(coherence_df.to_string(index=False))

# Cross-intent similarity
print("\n🔍 Cross-Intent Similarity (Lower = Better Separation):")
cross_intent_sim = []

for i, intent1 in enumerate(intent_labels[:5]):
    for intent2 in intent_labels[i+1:6]:
        samples1 = X_test[y_test == intent1].head(5)
        samples2 = X_test[y_test == intent2].head(5)
        
        if len(samples1) > 0 and len(samples2) > 0:
            emb1 = sem_model.encode(samples1.tolist())
            emb2 = sem_model.encode(samples2.tolist())
            
            sim = cosine_similarity(emb1, emb2).mean()
            cross_intent_sim.append({
                'Intent 1': intent1,
                'Intent 2': intent2,
                'Avg Similarity': sim
            })

cross_sim_df = pd.DataFrame(cross_intent_sim).sort_values('Avg Similarity', ascending=False)
print(cross_sim_df.head(10).to_string(index=False))

# -------------------------------
# 16. MISCLASSIFIED EXAMPLES
# -------------------------------
print("\n" + "="*60)
print("❌ MISCLASSIFIED EXAMPLES")
print("="*60)

mis_idx = np.where(y_test.values != y_pred_best)[0]
print(f"\nTotal misclassified: {len(mis_idx)} out of {len(y_test)} ({len(mis_idx)/len(y_test)*100:.2f}%)")

print("\nSample misclassified examples:")
for i in mis_idx[:10]:
    print(f"\nText: {X_test.iloc[i][:100]}...")
    print(f"True: {y_test.iloc[i]} → Predicted: {y_pred_best[i]}")

# -------------------------------
# 17. SAVE BEST MODEL
# -------------------------------
if 'Fusion' in best_feature_type:
    bundle = {
        "semantic": sem_model,
        "tfidf": tfidf,
        "clf": best_model,
        "scaler_sem": scaler_sem_base,
        "scaler_tfidf": scaler_tfidf_base,
        "alpha": best_alpha,
        "beta": best_beta,
        "feature_type": "fusion"
    }
elif 'TF-IDF' in best_feature_type:
    bundle = {
        "tfidf": tfidf,
        "clf": best_model,
        "scaler_tfidf": scaler_tfidf_base,
        "feature_type": "tfidf"
    }
else:
    bundle = {
        "semantic": sem_model,
        "clf": best_model,
        "scaler_sem": scaler_sem_base,
        "feature_type": "semantic"
    }

with open("models/thai_intent_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n💾 Saved best model bundle to 'models/thai_intent_best_model.pkl'")
print("\n✅ Analysis complete!")