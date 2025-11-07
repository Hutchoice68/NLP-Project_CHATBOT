# ============================================
# Improved Thai Intent Classification (Fixed for Deployment)
# WITH BACK-TRANSLATION AUGMENTATION
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ✅ เพิ่มส่วนนี้: Import Back-Translation
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import pickle
import os

from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================
# ✅ BACK-TRANSLATION AUGMENTOR (NEW!)
# ============================================
class BackTranslationAugmentor:
    """Back-Translation สำหรับภาษาไทยด้วย Hugging Face Models (ฟรี!)"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("🔧 Loading Translation Models...")
        self.device = device
        
        # Thai to English
        print("   Loading Thai → English model...")
        self.th2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
        self.th2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en").to(device)
        
        # English to Thai
        print("   Loading English → Thai model...")
        self.en2th_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-th")
        self.en2th_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-th").to(device)
        
        print(f"✅ Models loaded on {device}")
    
    def translate(self, text, src_tokenizer, src_model, max_length=128):
        """Translate text"""
        inputs = src_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = src_model.generate(**inputs, max_length=max_length)
        
        return src_tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def back_translate(self, text):
        """Thai → English → Thai"""
        try:
            # Step 1: Thai → English
            en_text = self.translate(text, self.th2en_tokenizer, self.th2en_model)
            
            # Step 2: English → Thai
            back_th = self.translate(en_text, self.en2th_tokenizer, self.en2th_model)
            
            if back_th.strip() != text.strip() and len(back_th) > 0:
                return [back_th]
            else:
                return []
        except Exception as e:
            return []


# ============================================
# ✅ IMPROVED DATA AUGMENTATION (REPLACED!)
# ============================================
def augment_minority_classes_backtrans(X_train, y_train, min_samples=10, 
                                       n_aug_per_sample=1, use_cache=True):
    """
    Augment minority classes ด้วย Back-Translation
    
    Parameters:
    -----------
    X_train : pd.Series - ข้อความ
    y_train : pd.Series - labels
    min_samples : int - threshold สำหรับ minority class
    n_aug_per_sample : int - จำนวน augmentation ต่อข้อความ
    use_cache : bool - บันทึกผลลัพธ์เพื่อไม่ต้องแปลซ้ำ
    
    Returns:
    --------
    X_augmented, y_augmented
    """
    # Initialize augmentor
    augmentor = BackTranslationAugmentor()
    
    # Load cache
    cache_file = '../models/augmentation_cache.pkl'
    cache = {}
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        print(f"📦 Loaded {len(cache)} cached translations")
    
    class_counts = Counter(y_train)
    minority_classes = [cls for cls, cnt in class_counts.items() if cnt < min_samples]
    
    if len(minority_classes) == 0:
        print("✅ No minority classes need augmentation")
        return X_train, y_train
    
    print(f"\n🔄 Augmenting {len(minority_classes)} minority classes with Back-Translation...")
    print(f"   Classes: {minority_classes}")
    
    X_aug_list = []
    y_aug_list = []
    
    for cls in minority_classes:
        cls_texts = X_train[y_train == cls].tolist()
        original_count = len(cls_texts)
        
        print(f"\n   Processing {cls} ({original_count} samples)...")
        
        for text in tqdm(cls_texts, desc=f"   {cls}", leave=False):
            # Check cache first
            if text in cache:
                augmented = cache[text]
            else:
                # Perform back-translation
                augmented = augmentor.back_translate(text)
                
                # Cache result
                if use_cache:
                    cache[text] = augmented
            
            # Add to augmented list (limit to n_aug_per_sample)
            X_aug_list.extend(augmented[:n_aug_per_sample])
            y_aug_list.extend([cls] * min(len(augmented), n_aug_per_sample))
        
        new_count = original_count + len([y for y in y_aug_list if y == cls])
        print(f"   ✅ {cls}: {original_count} → {new_count} samples")
    
    # Save cache
    if use_cache:
        os.makedirs('../models', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"\n💾 Saved {len(cache)} translations to cache")
    
    # Combine original + augmented
    X_combined = pd.concat([X_train, pd.Series(X_aug_list)], ignore_index=True)
    y_combined = pd.concat([y_train, pd.Series(y_aug_list)], ignore_index=True)
    
    print(f"\n📊 Final dataset: {len(X_train)} → {len(X_combined)} samples (+{len(X_aug_list)})")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return X_combined, y_combined


# ============================================
# EXISTING FUNCTIONS (เหมือนเดิม)
# ============================================
def smart_train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """แบ่งข้อมูลเป็น train/val/test แบบ stratified"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), 
        stratify=y, random_state=random_state
    )
    
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_optimal_fusion_weights(tfidf_features, semantic_features, y):
    """หาน้ำหนักที่ดีที่สุดด้วย validation"""
    best_score = 0
    best_alpha = 0.5
    
    for alpha in np.arange(0.3, 0.8, 0.1):
        beta = 1 - alpha
        fused = np.concatenate([
            semantic_features * alpha,
            tfidf_features * beta
        ], axis=1)
        
        clf = LogisticRegression(max_iter=200, random_state=42)
        scores = cross_val_score(clf, fused, y, cv=3, scoring='f1_weighted')
        score = scores.mean()
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    best_beta = 1 - best_alpha
    print(f"✅ Best fusion weights: α={best_alpha:.2f}, β={best_beta:.2f}")
    return best_alpha, best_beta


def fuse_features(tfidf_features, semantic_features, alpha, beta):
    """ผสม features ด้วยน้ำหนักที่กำหนด"""
    return np.concatenate([
        semantic_features * alpha,
        tfidf_features * beta
    ], axis=1)


def get_regularized_models():
    """โมเดลที่มี regularization ที่เหมาะสม"""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,  # ✅ ลด regularization (เดิมคือ 0.5)
            class_weight='balanced',
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,  # ✅ เพิ่มความยืดหยุ่น (เดิมคือ 8)
            min_samples_split=10,  # ✅ ลดข้อจำกัด (เดิมคือ 15)
            min_samples_leaf=5,  # ✅ ลดข้อจำกัด (เดิมคือ 8)
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf',
            C=0.5,  # ✅ เพิ่มจาก 0.3
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }


def detect_overfitting(train_scores, val_scores, threshold=0.1):
    """ตรวจจับ overfitting จากความต่างของ train/val"""
    gap = train_scores - val_scores
    if gap > threshold:
        print(f"⚠️ OVERFITTING DETECTED! Train-Val gap: {gap:.4f}")
        return True
    return False


# ============================================
# MAIN PIPELINE WITH BACK-TRANSLATION
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 IMPROVED THAI INTENT CLASSIFICATION")
    print("   WITH BACK-TRANSLATION AUGMENTATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv('../Data/intents_dataset.csv')
    print(f"✅ Loaded dataset: {df.shape}")
    
    # Preprocessing
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
    df['tokens'] = df['clean_text'].apply(
        lambda x: tokenize_with_stopwords(x, remove_stopwords=False)
    )
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    X = df['tokens']
    y = df['intent']
    
    # Better split: train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = smart_train_val_test_split(
        X, y, train_size=0.65, val_size=0.15, test_size=0.2  # ✅ ปรับเพิ่ม train size
    )
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # ✅ NEW: Back-Translation Augmentation
    print("\n🔄 Checking class balance...")
    class_counts = Counter(y_train)
    print(f"   Class distribution: {dict(class_counts)}")
    min_samples = min(class_counts.values())
    
    if min_samples < 10:
        X_train, y_train = augment_minority_classes_backtrans(
            X_train, y_train,
            min_samples=10,  # augment if class has < 10 samples
            n_aug_per_sample=1,
            use_cache=True
        )
    else:
        print("✅ All classes have sufficient samples (≥10)")
    
    # Create features
    print("\n🔧 Creating features...")
    sem_model = SentenceTransformer('sentence-transformers/LaBSE')
    
    emb_train = sem_model.encode(X_train.tolist(), show_progress_bar=True)
    emb_val = sem_model.encode(X_val.tolist(), show_progress_bar=False)
    emb_test = sem_model.encode(X_test.tolist(), show_progress_bar=False)
    
    tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
    tfidf_train = tfidf.fit_transform(X_train).toarray()
    tfidf_val = tfidf.transform(X_val).toarray()
    tfidf_test = tfidf.transform(X_test).toarray()
    
    # Scale features
    scaler_tfidf = StandardScaler()
    scaler_sem = StandardScaler()
    
    tfidf_train = scaler_tfidf.fit_transform(tfidf_train)
    tfidf_val = scaler_tfidf.transform(tfidf_val)
    tfidf_test = scaler_tfidf.transform(tfidf_test)
    
    emb_train = scaler_sem.fit_transform(emb_train)
    emb_val = scaler_sem.transform(emb_val)
    emb_test = scaler_sem.transform(emb_test)
    
    # Find optimal fusion weights
    print("\n🔬 Learning optimal fusion weights...")
    alpha, beta = find_optimal_fusion_weights(tfidf_val, emb_val, y_val)
    
    train_fused = fuse_features(tfidf_train, emb_train, alpha, beta)
    val_fused = fuse_features(tfidf_val, emb_val, alpha, beta)
    test_fused = fuse_features(tfidf_test, emb_test, alpha, beta)
    
    # Train models
    print("\n🎯 Training models...")
    models = get_regularized_models()
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔍 Evaluating: {name}")
        print(f"{'='*60}")
        
        model.fit(train_fused, y_train)
        
        # Evaluate
        y_val_pred = model.predict(val_fused)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = precision_recall_fscore_support(y_val, y_val_pred, average='weighted')[2]
        
        y_train_pred = model.predict(train_fused)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")
        
        detect_overfitting(train_acc, val_acc, threshold=0.15)
        
        if val_f1 > best_score:
            best_score = val_f1
            best_model = model
            best_model_name = name
    
    # Final evaluation
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*60)
    
    y_test_pred = best_model.predict(test_fused)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')[2]
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Save model
    bundle = {
        "semantic": sem_model,
        "tfidf": tfidf,
        "clf": best_model,
        "scaler_sem": scaler_sem,
        "scaler_tfidf": scaler_tfidf,
        "alpha": alpha,
        "beta": beta
    }
    
    with open("../models/improved_thai_intent.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print("\n💾 Model saved successfully!")
    print("✅ Training complete with Back-Translation Augmentation!")
    print(f"\n📝 Model components:")
    print(f"   - Semantic: LaBSE")
    print(f"   - TF-IDF: {tfidf.max_features} features")
    print(f"   - Classifier: {best_model_name}")
    print(f"   - Fusion: α={alpha:.2f}, β={beta:.2f}")
    print(f"   - Augmentation: Back-Translation (cached)")