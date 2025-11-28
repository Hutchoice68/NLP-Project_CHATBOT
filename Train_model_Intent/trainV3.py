# ============================================
# Improved Thai Intent Classification (Fixed for Deployment)
# WITH BACK-TRANSLATION AUGMENTATION + K-FOLD CV + LOGGING
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

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import pickle
import os
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ✅ สร้าง log directory
os.makedirs('../logs', exist_ok=True)
os.makedirs('../models2', exist_ok=True)


# ============================================
# ✅ LOGGING UTILITIES
# ============================================
class ExperimentLogger:
    """Logger สำหรับบันทึกทุกขั้นตอนของการทดลอง"""
    
    def __init__(self, log_dir='../logs'):
        self.log_dir = log_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # สร้าง log files แยกตามหัวข้อ
        self.main_log = os.path.join(log_dir, f'experiment_{timestamp}.log')
        self.stopword_log = os.path.join(log_dir, f'stopword_removal_{timestamp}.log')
        self.augmentation_log = os.path.join(log_dir, f'augmentation_{timestamp}.log')
        self.results_log = os.path.join(log_dir, f'model_results_{timestamp}.log')
        
        # เขียน header
        self._write_log(self.main_log, "="*80)
        self._write_log(self.main_log, "THAI INTENT CLASSIFICATION EXPERIMENT LOG")
        self._write_log(self.main_log, f"Started at: {datetime.now()}")
        self._write_log(self.main_log, "="*80 + "\n")
    
    def _write_log(self, filepath, message):
        """เขียน log ไปยังไฟล์"""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
        print(message)
    
    def log_main(self, message):
        """Log ข้อความหลัก"""
        self._write_log(self.main_log, message)
    
    def log_stopword(self, original, processed, removed_words):
        """Log การลบ stopwords"""
        log_msg = f"\nOriginal: {original}\n"
        log_msg += f"Processed: {processed}\n"
        log_msg += f"Removed: {removed_words}\n"
        log_msg += "-"*60
        self._write_log(self.stopword_log, log_msg)
    
    def log_augmentation(self, original, augmented, intent):
        """Log การ augment ข้อความ"""
        log_msg = f"\nIntent: {intent}\n"
        log_msg += f"Original:  {original}\n"
        log_msg += f"Augmented: {augmented}\n"
        log_msg += "-"*60
        self._write_log(self.augmentation_log, log_msg)
    
    def log_results(self, message):
        """Log ผลลัพธ์โมเดล"""
        self._write_log(self.results_log, message)


# ============================================
# ✅ BACK-TRANSLATION AUGMENTOR (WITH LOGGING)
# ============================================
class BackTranslationAugmentor:
    """Back-Translation สำหรับภาษาไทยด้วย Hugging Face Models (ฟรี!)"""
    
    def __init__(self, logger=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("🔧 Loading Translation Models...")
        self.device = device
        self.logger = logger
        
        # Thai to English
        print("   Loading Thai → English model...")
        self.th2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
        self.th2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en").to(device)
        
        # English to Thai
        print("   Loading English → Thai model...")
        self.en2th_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-th")
        self.en2th_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-th").to(device)
        
        print(f"✅ Models loaded on {device}")
        if self.logger:
            self.logger.log_main(f"Translation models loaded on {device}")
    
    def translate(self, text, src_tokenizer, src_model, max_length=128):
        """Translate text"""
        inputs = src_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = src_model.generate(**inputs, max_length=max_length)
        
        return src_tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def back_translate(self, text, intent=None):
        """Thai → English → Thai"""
        try:
            # Step 1: Thai → English
            en_text = self.translate(text, self.th2en_tokenizer, self.th2en_model)
            
            # Step 2: English → Thai
            back_th = self.translate(en_text, self.en2th_tokenizer, self.en2th_model)
            
            if back_th.strip() != text.strip() and len(back_th) > 0:
                # Log augmentation
                if self.logger and intent:
                    self.logger.log_augmentation(text, back_th, intent)
                return [back_th]
            else:
                return []
        except Exception as e:
            if self.logger:
                self.logger.log_main(f"Augmentation error: {e}")
            return []


# ============================================
# ✅ DATA AUGMENTATION (WITH LOGGING)
# ============================================
def augment_minority_classes_backtrans(X_train, y_train, logger=None, min_samples=10, 
                                       n_aug_per_sample=1, use_cache=True):
    """Augment minority classes ด้วย Back-Translation + Logging"""
    
    # Initialize augmentor
    augmentor = BackTranslationAugmentor(logger=logger)
    
    # Load cache
    cache_file = '../models2/augmentation_cache.pkl'
    cache = {}
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        msg = f"📦 Loaded {len(cache)} cached translations"
        print(msg)
        if logger:
            logger.log_main(msg)
    
    class_counts = Counter(y_train)
    minority_classes = [cls for cls, cnt in class_counts.items() if cnt < min_samples]
    
    if len(minority_classes) == 0:
        msg = "✅ No minority classes need augmentation"
        print(msg)
        if logger:
            logger.log_main(msg)
        return X_train, y_train
    
    msg = f"\n🔄 Augmenting {len(minority_classes)} minority classes with Back-Translation..."
    print(msg)
    if logger:
        logger.log_main(msg)
        logger.log_main(f"Classes: {minority_classes}")
    
    X_aug_list = []
    y_aug_list = []
    
    for cls in minority_classes:
        cls_texts = X_train[y_train == cls].tolist()
        original_count = len(cls_texts)
        
        msg = f"\n   Processing {cls} ({original_count} samples)..."
        print(msg)
        if logger:
            logger.log_main(msg)
        
        for text in tqdm(cls_texts, desc=f"   {cls}", leave=False):
            # Check cache first
            if text in cache:
                augmented = cache[text]
            else:
                # Perform back-translation (with logging)
                augmented = augmentor.back_translate(text, intent=cls)
                
                # Cache result
                if use_cache:
                    cache[text] = augmented
            
            # Add to augmented list
            X_aug_list.extend(augmented[:n_aug_per_sample])
            y_aug_list.extend([cls] * min(len(augmented), n_aug_per_sample))
        
        new_count = original_count + len([y for y in y_aug_list if y == cls])
        msg = f"   ✅ {cls}: {original_count} → {new_count} samples"
        print(msg)
        if logger:
            logger.log_main(msg)
    
    # Save cache
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        msg = f"\n💾 Saved {len(cache)} translations to cache"
        print(msg)
        if logger:
            logger.log_main(msg)
    
    # Combine original + augmented
    X_combined = pd.concat([X_train, pd.Series(X_aug_list)], ignore_index=True)
    y_combined = pd.concat([y_train, pd.Series(y_aug_list)], ignore_index=True)
    
    msg = f"\n📊 Final dataset: {len(X_train)} → {len(X_combined)} samples (+{len(X_aug_list)})"
    print(msg)
    if logger:
        logger.log_main(msg)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return X_combined, y_combined


# ============================================
# EXISTING FUNCTIONS
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
            max_iter=100,
            C=1.0,
            class_weight='balanced',
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf',
            C=0.5,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }


# ============================================
# ✅ K-FOLD CROSS-VALIDATION EVALUATION
# ============================================
def evaluate_with_kfold(model, X, y, n_splits=5, logger=None):
    """ประเมินโมเดลด้วย K-Fold Cross-Validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_val_fold)
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_fold, y_pred, average='weighted', zero_division=0
        )
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    results = {
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'accuracy_scores': accuracy_scores,
        'f1_scores': f1_scores
    }
    
    return results


# ============================================
# ✅ MODEL COMPARISON WITH K-FOLD
# ============================================
def comprehensive_model_comparison_kfold(models_dict, features_dict, y_train, y_val, y_test, 
                                         logger=None, n_splits=5):
    """เปรียบเทียบโมเดลด้วย K-Fold CV"""
    results = []
    
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE MODEL COMPARISON (K-FOLD CROSS-VALIDATION)")
    print("="*80)
    
    if logger:
        logger.log_results("\n" + "="*80)
        logger.log_results("MODEL COMPARISON WITH K-FOLD CV")
        logger.log_results("="*80)
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"📊 Testing: {model_name}")
        print(f"{'='*80}")
        
        if logger:
            logger.log_results(f"\n{'='*80}")
            logger.log_results(f"Model: {model_name}")
            logger.log_results(f"{'='*80}")
        
        for feature_name, (X_train, X_val, X_test) in features_dict.items():
            print(f"\n   🔍 Feature: {feature_name}")
            
            # Combine train + val for K-Fold
            X_combined = np.vstack([X_train, X_val])
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            
            # K-Fold CV
            kfold_results = evaluate_with_kfold(model, X_combined, y_combined, 
                                               n_splits=n_splits, logger=logger)
            
            # Final test evaluation
            model.fit(X_combined, y_combined)
            y_test_pred = model.predict(X_test)
            
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average='weighted', zero_division=0
            )
            
            # Store results
            result = {
                'model': model_name,
                'feature': feature_name,
                'cv_accuracy_mean': kfold_results['accuracy_mean'],
                'cv_accuracy_std': kfold_results['accuracy_std'],
                'cv_precision_mean': kfold_results['precision_mean'],
                'cv_precision_std': kfold_results['precision_std'],
                'cv_recall_mean': kfold_results['recall_mean'],
                'cv_recall_std': kfold_results['recall_std'],
                'cv_f1_mean': kfold_results['f1_mean'],
                'cv_f1_std': kfold_results['f1_std'],
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall
            }
            results.append(result)
            
            # Print results
            print(f"      CV Accuracy:  {kfold_results['accuracy_mean']:.4f} ± {kfold_results['accuracy_std']:.4f}")
            print(f"      CV F1-Score:  {kfold_results['f1_mean']:.4f} ± {kfold_results['f1_std']:.4f}")
            print(f"      CV Precision:  {kfold_results['precision_mean']:.4f} ± {kfold_results['precision_std']:.4f}")
            print(f"      CV Recall:  {kfold_results['recall_mean']:.4f} ± {kfold_results['recall_std']:.4f}")
            print(f"      Test Accuracy: {test_acc:.4f}")
            print(f"      Test F1-Score: {test_f1:.4f}")
            
            if logger:
                logger.log_results(f"\nFeature: {feature_name}")
                logger.log_results(f"CV Accuracy:  {kfold_results['accuracy_mean']:.4f} ± {kfold_results['accuracy_std']:.4f}")
                logger.log_results(f"CV F1-Score:  {kfold_results['f1_mean']:.4f} ± {kfold_results['f1_std']:.4f}")
                logger.log_results(f"Test Accuracy: {test_acc:.4f}")
                logger.log_results(f"Test F1-Score: {test_f1:.4f}")
    
    return pd.DataFrame(results)


# ============================================
# ✅ PLOT CONFUSION MATRICES
# ============================================
def plot_all_confusion_matrices(models_dict, features_dict, y_test, save_dir='../models2'):
    """สร้าง confusion matrix สำหรับทุกโมเดลและ feature combination"""
    
    print("\n📊 Generating confusion matrices...")
    
    n_models = len(models_dict)
    n_features = len(features_dict)
    
    fig, axes = plt.subplots(n_models, n_features, figsize=(6*n_features, 5*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (model_name, model) in enumerate(models_dict.items()):
        for j, (feature_name, (X_train, X_val, X_test)) in enumerate(features_dict.items()):
            
            # Train and predict
            X_combined = np.vstack([X_train, X_val])
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            model.fit(X_combined, y_combined)
            y_pred = model.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            ax = axes[i, j]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=model.classes_, 
                       yticklabels=model.classes_,
                       cbar=True, ax=ax)
            
            ax.set_title(f'{model_name}\n{feature_name}', fontsize=10, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrices saved to: {save_path}")
    plt.show()


# ============================================
# ✅ VISUALIZATION FUNCTIONS
# ============================================
def plot_comparison_results(results_df, save_path='../models2/comparison_plots.png'):
    """สร้างกราฟเปรียบเทียบผลลัพธ์"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison (K-Fold CV)', fontsize=16, fontweight='bold')
    
    # 1. CV F1-Score with error bars
    pivot_f1 = results_df.pivot_table(values='cv_f1_mean', index='model', columns='feature')
    pivot_f1_std = results_df.pivot_table(values='cv_f1_std', index='model', columns='feature')
    
    x = np.arange(len(pivot_f1.index))
    width = 0.25
    
    for idx, col in enumerate(pivot_f1.columns):
        axes[0, 0].bar(x + idx*width, pivot_f1[col], width, 
                      yerr=pivot_f1_std[col], label=col, capsize=5)
    
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('Cross-Validation F1-Score (Mean ± Std)')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(pivot_f1.index, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Test vs CV F1-Score
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        axes[0, 1].scatter(model_data['cv_f1_mean'], model_data['test_f1'], 
                          s=100, alpha=0.6, label=model)
        
        for idx, row in model_data.iterrows():
            axes[0, 1].annotate(row['feature'], 
                               (row['cv_f1_mean'], row['test_f1']),
                               fontsize=7, alpha=0.7)
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 1].set_xlabel('CV F1-Score (Mean)')
    axes[0, 1].set_ylabel('Test F1-Score')
    axes[0, 1].set_title('CV vs Test Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Test Accuracy Comparison
    pivot_test_acc = results_df.pivot_table(values='test_acc', index='model', columns='feature')
    pivot_test_acc.plot(kind='bar', ax=axes[1, 0], rot=45)
    axes[1, 0].set_title('Test Accuracy by Model & Feature')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend(title='Feature Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Variance Analysis
    results_df['cv_variance'] = results_df['cv_f1_std']
    pivot_var = results_df.pivot_table(values='cv_variance', index='model', columns='feature')
    pivot_var.plot(kind='bar', ax=axes[1, 1], rot=45, color=['#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 1].set_title('Model Stability (CV F1-Score Std Dev)')
    axes[1, 1].set_ylabel('Standard Deviation (Lower is Better)')
    axes[1, 1].legend(title='Feature Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Comparison plots saved to: {save_path}")
    plt.show()


def print_best_models(results_df, logger=None):
    """แสดงโมเดลที่ดีที่สุดในแต่ละหมวด"""
    summary = "\n" + "="*80 + "\n"
    summary += "🏆 BEST MODELS SUMMARY (K-FOLD CV)\n"
    summary += "="*80 + "\n"
    
    # Best overall (by CV F1)
    best_overall = results_df.loc[results_df['cv_f1_mean'].idxmax()]
    summary += f"\n🥇 BEST OVERALL (CV F1-Score):\n"
    summary += f"   Model: {best_overall['model']}\n"
    summary += f"   Feature: {best_overall['feature']}\n"
    summary += f"   CV F1: {best_overall['cv_f1_mean']:.4f} ± {best_overall['cv_f1_std']:.4f}\n"
    summary += f"   Test F1: {best_overall['test_f1']:.4f}\n"
    
    # Most stable
    most_stable = results_df.loc[results_df['cv_f1_std'].idxmin()]
    summary += f"\n🎯 MOST STABLE MODEL (Lowest Variance):\n"
    summary += f"   Model: {most_stable['model']}\n"
    summary += f"   Feature: {most_stable['feature']}\n"
    summary += f"   CV F1 Std: {most_stable['cv_f1_std']:.4f}\n"
    summary += f"   CV F1 Mean: {most_stable['cv_f1_mean']:.4f}\n"
    
    print(summary)
    if logger:
        logger.log_results(summary)


# ============================================
# MAIN PIPELINE
# ============================================
if __name__ == "__main__":
    # Initialize logger
    logger = ExperimentLogger()
    
    print("="*80)
    print("🚀 IMPROVED THAI INTENT CLASSIFICATION")
    print("   WITH K-FOLD CV + BACK-TRANSLATION + LOGGING")
    print("="*80)
    logger.log_main("="*80)
    logger.log_main("EXPERIMENT STARTED")
    logger.log_main("="*80)
    
    # Load data
    df = pd.read_csv('D:\inten_nlp\Train_model_Intent\Data\intents_dataset_diverse.csv')
    msg = f"✅ Loaded dataset: {df.shape}"
    print(msg)
    logger.log_main(msg)
    
    # Preprocessing with logging
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r"[^ก-๙a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def tokenize_with_stopwords(text, remove_stopwords=False, log_sample=False):
        tokens = word_tokenize(text, engine='newmm')
        
        if remove_stopwords:
            stop_words = set(thai_stopwords())
            removed = [t for t in tokens if t in stop_words]
            tokens = [t for t in tokens if t not in stop_words]
            
            # Log samples (first 5 texts)
            if log_sample and len(removed) > 0:
                logger.log_stopword(text, " ".join(tokens), removed)
        
        return " ".join(tokens)
    
    logger.log_main("\n📝 Preprocessing data...")
    logger.log_main("Stopword removal samples (first 5 texts):")
    
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Log first 5 stopword removals
    for idx in range(min(5, len(df))):
        text = df.iloc[idx]['clean_text']
        tokenize_with_stopwords(text, remove_stopwords=True, log_sample=True)
    
    df['tokens'] = df['clean_text'].apply(
        lambda x: tokenize_with_stopwords(x, remove_stopwords=False)
    )
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    X = df['tokens']
    y = df['intent']
    
    msg = f"✅ Preprocessing complete. Final dataset: {len(df)} samples"
    print(msg)
    logger.log_main(msg)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = smart_train_val_test_split(
        X, y, train_size=0.65, val_size=0.15, test_size=0.2
    )
    
    split_msg = f"\n📊 Data split:\n"
    split_msg += f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)\n"
    split_msg += f"   Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)\n"
    split_msg += f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)"
    print(split_msg)
    logger.log_main(split_msg)
    
    # Back-Translation Augmentation
    print("\n🔄 Checking class balance...")
    logger.log_main("\n🔄 AUGMENTATION PHASE")
    
    class_counts = Counter(y_train)
    balance_msg = f"   Class distribution: {dict(class_counts)}"
    print(balance_msg)
    logger.log_main(balance_msg)
    
    min_samples = min(class_counts.values())
    
    if min_samples < 100:
        X_train, y_train = augment_minority_classes_backtrans(
            X_train, y_train,
            logger=logger,
            min_samples=10,
            n_aug_per_sample=1,
            use_cache=True
        )
    else:
        msg = "✅ All classes have sufficient samples (≥10)"
        print(msg)
        logger.log_main(msg)
    
    # Create features
    print("\n🔧 Creating features...")
    logger.log_main("\n🔧 FEATURE ENGINEERING")
    
    sem_model = SentenceTransformer('sentence-transformers/LaBSE')
    logger.log_main("   - Semantic model: LaBSE")
    
    emb_train = sem_model.encode(X_train.tolist(), show_progress_bar=True)
    emb_val = sem_model.encode(X_val.tolist(), show_progress_bar=False)
    emb_test = sem_model.encode(X_test.tolist(), show_progress_bar=False)
    
    tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
    tfidf_train = tfidf.fit_transform(X_train).toarray()
    tfidf_val = tfidf.transform(X_val).toarray()
    tfidf_test = tfidf.transform(X_test).toarray()
    logger.log_main(f"   - TF-IDF: {tfidf.max_features} features")
    
    # Scale features
    scaler_tfidf = StandardScaler(with_mean=False)
    scaler_sem = StandardScaler()
    
    tfidf_train_scaled = scaler_tfidf.fit_transform(tfidf_train)
    tfidf_val_scaled = scaler_tfidf.transform(tfidf_val)
    tfidf_test_scaled = scaler_tfidf.transform(tfidf_test)
    
    emb_train_scaled = scaler_sem.fit_transform(emb_train)
    emb_val_scaled = scaler_sem.transform(emb_val)
    emb_test_scaled = scaler_sem.transform(emb_test)
    
    # Find optimal fusion weights
    print("\n🔬 Learning optimal fusion weights...")
    alpha, beta = find_optimal_fusion_weights(tfidf_val_scaled, emb_val_scaled, y_val)
    logger.log_main(f"\n🔬 Optimal fusion weights: α={alpha:.2f}, β={beta:.2f}")
    
    train_fused = fuse_features(tfidf_train_scaled, emb_train_scaled, alpha, beta)
    val_fused = fuse_features(tfidf_val_scaled, emb_val_scaled, alpha, beta)
    test_fused = fuse_features(tfidf_test_scaled, emb_test_scaled, alpha, beta)
    
    # Prepare feature sets
    features_dict = {
        "TF-IDF Only": (tfidf_train_scaled, tfidf_val_scaled, tfidf_test_scaled),
        "Sentence-BERT Only": (emb_train_scaled, emb_val_scaled, emb_test_scaled),
        "Fusion (Weighted)": (train_fused, val_fused, test_fused)
    }
    
    # Get models
    models_dict = get_regularized_models()
    logger.log_main(f"\n🤖 Models to evaluate: {list(models_dict.keys())}")
    logger.log_main(f"📊 Feature types: {list(features_dict.keys())}")
    logger.log_main(f"🔄 K-Fold splits: 5")
    
    # Run comprehensive comparison with K-Fold
    results_df = comprehensive_model_comparison_kfold(
        models_dict, features_dict, y_train, y_val, y_test, 
        logger=logger, n_splits=5
    )
    
    # Save results
    results_path = '../models2/model_comparison_kfold_results.csv'
    results_df.to_csv(results_path, index=False)
    msg = f"\n💾 Results saved to: {results_path}"
    print(msg)
    logger.log_main(msg)
    
    # Display results table
    print("\n" + "="*80)
    print("📋 FULL COMPARISON TABLE (K-FOLD CV)")
    print("="*80)
    display_cols = ['model', 'feature', 'cv_f1_mean', 'cv_f1_std', 
                    'test_f1', 'cv_accuracy_mean', 'cv_accuracy_std', 'test_acc']
    print(results_df[display_cols].to_string(index=False))
    
    logger.log_results("\n" + "="*80)
    logger.log_results("FULL RESULTS TABLE")
    logger.log_results("="*80)
    logger.log_results(results_df[display_cols].to_string(index=False))
    
    # Print best models
    print_best_models(results_df, logger)
    
    # Plot comparison
    plot_comparison_results(results_df)
    
    # Plot confusion matrices
    plot_all_confusion_matrices(models_dict, features_dict, y_test)
    
    # Save best model
    best_row = results_df.loc[results_df['cv_f1_mean'].idxmax()]
    best_model_name = best_row['model']
    best_feature_type = best_row['feature']
    
    print(f"\n💾 Saving best model: {best_model_name} with {best_feature_type}")
    logger.log_main(f"\n💾 Best model selected: {best_model_name} with {best_feature_type}")
    
    # Re-train best model on train+val
    best_model = models_dict[best_model_name]
    if best_feature_type == "TF-IDF Only":
        X_train_best = np.vstack([tfidf_train_scaled, tfidf_val_scaled])
        X_test_best = tfidf_test_scaled
    elif best_feature_type == "Sentence-BERT Only":
        X_train_best = np.vstack([emb_train_scaled, emb_val_scaled])
        X_test_best = emb_test_scaled
    else:  # Fusion
        X_train_best = np.vstack([train_fused, val_fused])
        X_test_best = test_fused
    
    y_train_best = pd.concat([y_train, y_val], ignore_index=True)
    best_model.fit(X_train_best, y_train_best)
    
    # Final evaluation
    print("\n" + "="*80)
    print(f"🎯 DETAILED CLASSIFICATION REPORT - BEST MODEL")
    print(f"   ({best_model_name} + {best_feature_type})")
    print("="*80)
    
    y_test_pred = best_model.predict(X_test_best)
    report = classification_report(y_test, y_test_pred)
    print(report)
    
    logger.log_results("\n" + "="*80)
    logger.log_results(f"BEST MODEL CLASSIFICATION REPORT")
    logger.log_results(f"Model: {best_model_name} + {best_feature_type}")
    logger.log_results("="*80)
    logger.log_results(report)
    
    # Individual confusion matrix for best model
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, 
                yticklabels=best_model.classes_)
    plt.title(f'Best Model: {best_model_name} ({best_feature_type})', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    best_cm_path = '../models2/best_model_confusion_matrix.png'
    plt.savefig(best_cm_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Best model confusion matrix saved to: {best_cm_path}")
    plt.show()
    
    # Save model bundle
    bundle = {
        "semantic": sem_model,
        "tfidf": tfidf,
        "clf": best_model,
        "scaler_sem": scaler_sem,
        "scaler_tfidf": scaler_tfidf,
        "alpha": alpha,
        "beta": beta,
        "best_model_name": best_model_name,
        "best_feature_type": best_feature_type,
        "comparison_results": results_df,
        "kfold_results": {
            'cv_f1_mean': best_row['cv_f1_mean'],
            'cv_f1_std': best_row['cv_f1_std'],
            'test_f1': best_row['test_f1']
        }
    }
    
    model_path = "../models2/improved_thai_intent_kfold.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"\n💾 Model bundle saved to: {model_path}")
    logger.log_main(f"\n💾 Model bundle saved to: {model_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    summary = f"""
📦 Generated Files:
   1. {model_path} - Best model bundle
   2. {results_path} - Full comparison results (CSV)
   3. ../models2/comparison_plots.png - Visual comparison
   4. ../models2/all_confusion_matrices.png - All confusion matrices
   5. {best_cm_path} - Best model confusion matrix
   6. ../models2/augmentation_cache.pkl - Back-translation cache
   
📋 Log Files:
   1. {logger.main_log} - Main experiment log
   2. {logger.stopword_log} - Stopword removal samples
   3. {logger.augmentation_log} - Back-translation samples
   4. {logger.results_log} - Model evaluation results

💡 Best Configuration:
   • Model: {best_model_name}
   • Feature: {best_feature_type}
   • CV F1-Score: {best_row['cv_f1_mean']:.4f} ± {best_row['cv_f1_std']:.4f}
   • Test F1-Score: {best_row['test_f1']:.4f}
   • Test Accuracy: {best_row['test_acc']:.4f}

🔬 Evaluation Method:
   • 5-Fold Cross-Validation
   • Stratified splits
   • Train/Val/Test: 65%/15%/20%

📊 Total Experiments: {len(results_df)}
   • Models tested: {len(models_dict)}
   • Feature types: {len(features_dict)}
   • Metrics: Accuracy, F1, Precision, Recall (with std dev)
"""
    
    print(summary)
    logger.log_main("\n" + "="*80)
    logger.log_main("EXPERIMENT SUMMARY")
    logger.log_main("="*80)
    logger.log_main(summary)
    
    print("\n🔍 To analyze results:")
    print("   df = pd.read_csv('../models2/model_comparison_kfold_results.csv')")
    print("   df.sort_values('cv_f1_mean', ascending=False)")
    
    logger.log_main(f"\nExperiment completed at: {datetime.now()}")
    print(f"\n✅ All logs saved to: {logger.log_dir}")
    print("="*80)