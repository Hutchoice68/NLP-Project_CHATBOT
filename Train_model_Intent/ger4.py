"""
================================================================================
THAI INTENT DATASET GENERATOR - SEQUENTIAL AUGMENTATION PIPELINE
================================================================================
Pipeline Flow:
    Original Sentence
         ↓
    Back-Translation (th→en→th)
         ↓
    Thai2Fit fastText (synonym replacement)
         ↓
    Thai2Fit ULMFit (contextual generation)
         ↓
    Filter Duplicates + Check Intent Consistency
================================================================================
"""

import pandas as pd
import random
import string
import re
import os
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import nltk
# ============================================================================
# 1. BACKTRANSLATION MODULE
# ============================================================================

class BacktranslateAugmenter:
    """Step 1: Back-translation (th→en→th)"""
    
    def __init__(self, use_gpu: bool = False):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch
            
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            print(f"🔄 [1/4] Loading Backtranslate models (device: {self.device})...")
            
            # Thai → English
            self.th2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
            self.th2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en").to(self.device)
            
            # English → Thai
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            print("   Using mBART50 for en→th translation...")
            self.en2th_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.en2th_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(self.device)
            self.use_mbart = True
            
            self.available = True
            print("   ✅ Backtranslate ready")
            
        except Exception as e:
            print(f"   ⚠️  Backtranslate unavailable: {e}")
            self.available = False
    
    def augment(self, text: str) -> str:
        """Back-translate: Thai → English → Thai"""
        if not self.available or len(text.strip()) < 3:
            return text
        
        try:
            # Thai → English
            inputs = self.th2en_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            translated = self.th2en_model.generate(**inputs, max_length=512)
            en_text = self.th2en_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # English → Thai
            inputs = self.en2th_tokenizer(en_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            translated = self.en2th_model.generate(**inputs, max_length=512)
            th_text = self.en2th_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return th_text.strip()
        except Exception as e:
            return text


# ============================================================================
# 2. THAI2FIT FASTTEXT MODULE (Synonym Replacement)
# ============================================================================

class Thai2FitFastText:
    """Step 2: Thai2Fit fastText synonym replacement"""
    
    def __init__(self):
        print("📚 [2/4] Loading Thai2Fit fastText (word embeddings)...")
        
        try:
            from pythainlp.word_vector import get_model
            from pythainlp import word_tokenize
            
            # ดาวน์โหลดและโหลดโมเดล Thai2Fit
            print("   📥 Downloading Thai2Fit model...")
            self.word2vec = get_model("thai2fit_wv")  # ชื่อใหม่
            self.tokenizer = word_tokenize
            self.available = True
            print("   ✅ Thai2Fit fastText ready")
            
        except Exception as e:
            print(f"   ⚠️  Thai2Fit fastText unavailable: {e}")
            print("   💡 Install: pip install pythainlp gensim")
            self.available = False
            
            # Fallback dictionary
            self.fallback_dict = {
                "ขอ": ["อยาก", "ต้องการ"], "อยาก": ["ขอ", "ต้องการ"],
                "ซื้อ": ["สั่ง", "จอง"], "สั่ง": ["ซื้อ", "จอง"],
                "สินค้า": ["ของ", "ไอเทม"], "ราคา": ["เท่าไหร่", "กี่บาท"],
                "ดี": ["เยี่ยม", "เจ๋ง"], "สวย": ["งาม", "น่ารัก"],
                "มาก": ["เลย", "จัง"], "หน่อย": ["นิดหน่อย", "บ้าง"],
            }
    
    def get_similar_words(self, word: str, topn: int = 5) -> List[str]:
        """หาคำใกล้เคียงจาก word embeddings"""
        if not self.available:
            return self.fallback_dict.get(word, [])
        
        try:
            similar = self.word2vec.most_similar(word, topn=topn)
            return [w[0] for w in similar if len(w[0]) >= 2]
        except:
            return self.fallback_dict.get(word, [])
    
    def augment(self, text: str, num_replacements: int = 2) -> str:
        """แทนที่คำด้วย synonym (ไม่เปลี่ยนความหมาย)"""
        if len(text.strip()) < 3:
            return text
        
        try:
            # Tokenize
            if self.available:
                tokens = self.tokenizer(text, engine='newmm')
            else:
                tokens = text.split()
            
            # เลือกคำที่จะแทนที่
            candidates = [i for i, t in enumerate(tokens) if len(t) >= 2 and t.isalpha()]
            if not candidates:
                return text
            
            # Random select tokens to replace
            num_to_replace = min(num_replacements, len(candidates))
            indices_to_replace = random.sample(candidates, num_to_replace)
            
            # Replace
            for idx in indices_to_replace:
                token = tokens[idx]
                similar_words = self.get_similar_words(token, topn=5)
                
                if similar_words:
                    # เลือกคำที่มีความยาวใกล้เคียง
                    suitable = [w for w in similar_words if abs(len(w) - len(token)) <= 3]
                    if suitable:
                        tokens[idx] = random.choice(suitable[:3])
            
            return ''.join(tokens) if self.available else ' '.join(tokens)
        except:
            return text


# ============================================================================
# 3. THAI2FIT ULMFIT MODULE (Contextual Generation)
# ============================================================================

class Thai2FitULMFit:
    """Step 3: Thai2Fit ULMFit contextual text generation"""
    
    def __init__(self):
        print("🧠 [3/4] Loading Thai2Fit ULMFit (language model)...")
        
        try:
            from pythainlp.augment.lm import Thai2fitAug
            
            self.augmenter = Thai2fitAug()
            self.available = True
            print("   ✅ Thai2Fit ULMFit ready")
            
        except Exception as e:
            print(f"   ⚠️  Thai2Fit ULMFit unavailable: {e}")
            print("   💡 This uses pythainlp's language model")
            self.available = False
    
    def augment(self, text: str) -> str:
        """Generate contextual variations"""
        if not self.available or len(text.strip()) < 5:
            return text
        
        try:
            # ULMFit จะพยายามสร้างประโยคใหม่ที่มีความหมายใกล้เคียง
            augmented = self.augmenter.augment(text, n_sent=1)
            return augmented[0] if augmented else text
        except:
            return text


# ============================================================================
# 4. QUALITY FILTER MODULE
# ============================================================================

class QualityFilter:
    """Step 4: Filter duplicates + Check intent consistency"""
    
    def __init__(self, intent_keywords: Dict[str, List[str]]):
        print("🔍 [4/4] Initializing Quality Filter...")
        self.intent_keywords = intent_keywords
        self.seen_texts = set()
        print("   ✅ Quality Filter ready")
    
    def normalize_text(self, text: str) -> str:
        """Normalize สำหรับการเปรียบเทียบ"""
        # ลบ whitespace ส่วนเกิน
        text = re.sub(r'\s+', ' ', text.strip())
        # ลบ punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # lowercase
        return text.lower()
    
    def is_duplicate(self, text: str, threshold: float = 0.9) -> bool:
        """ตรวจสอบ duplicate (exact match และ normalized)"""
        normalized = self.normalize_text(text)
        
        # Exact match
        if normalized in self.seen_texts:
            return True
        
        # Similarity check (optional - ใช้ Levenshtein distance)
        # สำหรับ demo นี้ใช้ exact match ก่อน
        
        self.seen_texts.add(normalized)
        return False
    
    def check_intent_consistency(self, text: str, intent: str) -> bool:
        """ตรวจสอบว่า augmented text ยังคง intent เดิมหรือไม่"""
        if intent not in self.intent_keywords:
            return True  # unknown intent ให้ผ่านไปก่อน
        
        keywords = self.intent_keywords[intent]
        text_lower = text.lower()
        
        # ตรวจสอบว่ามีคำสำคัญของ intent นี้หรือไม่
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        # ต้องมีอย่างน้อย 1 keyword หรือถ้า text สั้นมาก (< 5 chars) ให้ผ่าน
        if matches > 0 or len(text.strip()) < 5:
            return True
        
        return False
    
    def filter(self, text: str, intent: str) -> Tuple[bool, str]:
        """
        Filter augmented text
        Returns: (is_valid, reason)
        """
        # Check 1: Too short or too long
        if len(text.strip()) < 2:
            return False, "too_short"
        if len(text.strip()) > 500:
            return False, "too_long"
        
        # Check 2: Duplicate
        if self.is_duplicate(text):
            return False, "duplicate"
        
        # Check 3: Intent consistency
        if not self.check_intent_consistency(text, intent):
            return False, "intent_mismatch"
        
        # Check 4: Too many repeated characters (spam)
        if re.search(r'(.)\1{5,}', text):
            return False, "spam_chars"
        
        return True, "valid"


# ============================================================================
# 5. SEQUENTIAL AUGMENTATION PIPELINE
# ============================================================================

class SequentialAugmentationPipeline:
    """Main pipeline that chains all augmentation steps"""
    
    def __init__(
        self,
        use_backtranslate: bool = True,
        use_fasttext: bool = True,
        use_ulmfit: bool = True,
        use_gpu: bool = False
    ):
        print("\n" + "="*80)
        print("🚀 INITIALIZING SEQUENTIAL AUGMENTATION PIPELINE")
        print("="*80 + "\n")
        
        # Initialize augmenters
        self.backtranslate = BacktranslateAugmenter(use_gpu) if use_backtranslate else None
        self.fasttext = Thai2FitFastText() if use_fasttext else None
        self.ulmfit = Thai2FitULMFit() if use_ulmfit else None
        
        # Intent keywords for filtering
        self.intent_keywords = {
            "ask_info": ["ราคา", "ข้อมูล", "รายละเอียด", "สอบถาม", "อยากรู้", "spec", "โปร", "ส่วนลด"],
            "order_product": ["ซื้อ", "สั่ง", "จอง", "order", "cf", "เอา", "อยากได้"],
            "refund_request": ["คืน", "เงิน", "ของ", "ปัญหา", "เสีย", "ผิด", "refund", "เปลี่ยน"],
            "help_request": ["ช่วย", "help", "ปัญหา", "ติดต่อ", "แอดมิน", "error", "ไม่ได้"],
            "greeting": ["สวัสดี", "หวัด", "hello", "hi", "ดี"],
            "thank_you": ["ขอบคุณ", "ขอบใจ", "thank"],
            "feedback": ["ดี", "ชอบ", "แย่", "ประทับใจ", "บริการ", "สินค้า"],
        }
        
        self.filter = QualityFilter(self.intent_keywords)
        
        print("\n" + "="*80)
        print("✅ PIPELINE READY")
        print("="*80 + "\n")
    
    def augment_single(
        self,
        text: str,
        intent: str,
        apply_backtranslate: bool = True,
        apply_fasttext: bool = True,
        apply_ulmfit: bool = True
    ) -> List[Dict[str, str]]:
        """
        Augment single text through the pipeline
        
        Returns list of augmented variants with metadata
        """
        results = []
        current_text = text
        
        # Original
        results.append({
            "text": current_text,
            "stage": "original",
            "intent": intent
        })
        
        # Stage 1: Backtranslation
        if apply_backtranslate and self.backtranslate and self.backtranslate.available:
            bt_text = self.backtranslate.augment(current_text)
            if bt_text != current_text:
                is_valid, reason = self.filter.filter(bt_text, intent)
                if is_valid:
                    results.append({
                        "text": bt_text,
                        "stage": "backtranslate",
                        "intent": intent
                    })
                    current_text = bt_text  # Use for next stage
        
        # Stage 2: fastText (synonym replacement)
        if apply_fasttext and self.fasttext:
            ft_text = self.fasttext.augment(current_text, num_replacements=2)
            if ft_text != current_text:
                is_valid, reason = self.filter.filter(ft_text, intent)
                if is_valid:
                    results.append({
                        "text": ft_text,
                        "stage": "fasttext",
                        "intent": intent
                    })
                    current_text = ft_text  # Use for next stage
        
        # Stage 3: ULMFit (contextual generation)
        if apply_ulmfit and self.ulmfit and self.ulmfit.available:
            ulm_text = self.ulmfit.augment(current_text)
            if ulm_text != current_text:
                is_valid, reason = self.filter.filter(ulm_text, intent)
                if is_valid:
                    results.append({
                        "text": ulm_text,
                        "stage": "ulmfit",
                        "intent": intent
                    })
        
        return results
    
    def augment_batch(
        self,
        texts: List[str],
        intents: List[str],
        augment_prob: float = 0.5,
        max_variants_per_text: int = 3
    ) -> List[Dict[str, str]]:
        """
        Augment batch of texts
        
        Args:
            texts: List of original texts
            intents: List of corresponding intents
            augment_prob: Probability of augmenting each text
            max_variants_per_text: Maximum augmented variants per original
        """
        all_results = []
        
        for text, intent in zip(texts, intents):
            # Always include original
            all_results.append({
                "text": text,
                "stage": "original",
                "intent": intent
            })
            
            # Augment with probability
            if random.random() < augment_prob:
                # Randomly decide which stages to apply
                apply_bt = random.random() < 0.3  # 30% chance
                apply_ft = random.random() < 0.7  # 70% chance
                apply_ulm = random.random() < 0.4  # 40% chance
                
                augmented = self.augment_single(
                    text, intent,
                    apply_backtranslate=apply_bt,
                    apply_fasttext=apply_ft,
                    apply_ulmfit=apply_ulm
                )
                
                # Add augmented variants (skip original)
                variants = augmented[1:]  # Skip original
                if variants:
                    # Limit variants
                    selected = random.sample(variants, min(len(variants), max_variants_per_text))
                    all_results.extend(selected)
        
        return all_results


# ============================================================================
# 6. INTENT TEMPLATES
# ============================================================================

intents_templates = {
    "ask_info": [
        "ราคา", "ขอราคา", "เอาข้อมูล", "อยากรู้ราคา", "มีโปรไหม",
        "ขอรายละเอียดหน่อยครับ", "อยากทราบข้อมูลเกี่ยวกับสินค้านี้"
    ],
    "order_product": [
        "ซื้อ", "สั่ง", "ซื้อเลย", "CF ค่ะ", "อยากได้สินค้านี้",
        "ขอสั่งซื้อสินค้านี้หนึ่งชิ้นได้ไหมครับ"
    ],
    "refund_request": [
        "คืนเงิน", "ของพัง", "ขอคืนเงิน", "สินค้าเสีย", "อยากเปลี่ยนสินค้า",
        "อยากขอคืนเงินได้ไหมครับ สินค้ามีปัญหา"
    ],
    "help_request": [
        "ช่วย", "help", "ช่วยด้วย", "มีปัญหา", "ล็อกอินไม่ได้",
        "รบกวนช่วยเหลือหน่อยได้ไหมครับ"
    ],
    "greeting": [
        "สวัสดี", "สวัสดีครับ", "หวัดดี", "Hello"
    ],
    "thank_you": [
        "ขอบคุณ", "ขอบคุณครับ", "Thanks", "ขอบใจ"
    ],
    "feedback": [
        "ดีมาก", "ชอบ", "ของดี", "บริการดี", "ประทับใจ"
    ],
    "unknown": [
        "กินข้าวยัง", "ฝนตกหนัก", "555555", "asdfgh"
    ]
}


# ============================================================================
# 7. MAIN DATASET GENERATOR
# ============================================================================

def generate_sequential_augmented_dataset(
    target_total: int = 40000,
    augment_prob: float = 0.5,
    max_variants: int = 3,
    use_backtranslate: bool = True,
    use_fasttext: bool = True,
    use_ulmfit: bool = True,
    use_gpu: bool = False,
    save_path: str = "Data/intents_dataset_sequential.csv"
):
    """
    Generate dataset using sequential augmentation pipeline
    
    Args:
        target_total: Target number of samples
        augment_prob: Probability of augmenting each sample
        max_variants: Max augmented variants per original
        use_backtranslate: Enable backtranslation
        use_fasttext: Enable Thai2Fit fastText
        use_ulmfit: Enable Thai2Fit ULMFit
        use_gpu: Use GPU for models
        save_path: Output file path
    """
    
    print("\n" + "="*80)
    print("📊 GENERATING SEQUENTIAL AUGMENTED DATASET")
    print("="*80)
    print(f"\nTarget: {target_total:,} samples")
    print(f"Augmentation probability: {augment_prob:.1%}")
    print(f"Max variants per text: {max_variants}")
    print(f"Backtranslate: {'✓' if use_backtranslate else '✗'}")
    print(f"FastText: {'✓' if use_fasttext else '✗'}")
    print(f"ULMFit: {'✓' if use_ulmfit else '✗'}")
    print(f"GPU: {'✓' if use_gpu else '✗'}\n")
    
    # Initialize pipeline
    pipeline = SequentialAugmentationPipeline(
        use_backtranslate=use_backtranslate,
        use_fasttext=use_fasttext,
        use_ulmfit=use_ulmfit,
        use_gpu=use_gpu
    )
    
    # Generate base samples
    base_samples = []
    base_per_intent = target_total // (len(intents_templates) * (1 + max_variants))
    
    print(f"🔧 Generating base samples (~{base_per_intent} per intent)...\n")
    
    for intent, templates in intents_templates.items():
        for _ in range(base_per_intent):
            text = random.choice(templates)
            base_samples.append((text, intent))
    
    # Augment
    print(f"🔄 Running augmentation pipeline...\n")
    
    augmented_results = pipeline.augment_batch(
        texts=[t[0] for t in base_samples],
        intents=[t[1] for t in base_samples],
        augment_prob=augment_prob,
        max_variants_per_text=max_variants
    )
    
    # Create DataFrame
    df = pd.DataFrame(augmented_results)
    
    # Add metadata
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(lambda x: len(x))
    df['platform'] = random.choices(
        ["LINE", "Facebook", "Web", "Instagram"],
        k=len(df)
    )
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Limit to target
    if len(df) > target_total:
        df = df.iloc[:target_total]
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    # Statistics
    print("\n" + "="*80)
    print("✅ DATASET GENERATED")
    print("="*80)
    print(f"\n📁 Saved: {save_path}")
    print(f"📊 Total samples: {len(df):,}")
    
    print(f"\n📋 Intent Distribution:")
    print(df['intent'].value_counts().to_string())
    
    print(f"\n🔄 Augmentation Stage Distribution:")
    print(df['stage'].value_counts().to_string())
    
    print(f"\n📏 Length Statistics:")
    print(f"   Words: min={df['word_count'].min()}, max={df['word_count'].max()}, mean={df['word_count'].mean():.1f}")
    print(f"   Chars: min={df['char_count'].min()}, max={df['char_count'].max()}, mean={df['char_count'].mean():.1f}")
    
    # Show examples by stage
    print(f"\n🔍 EXAMPLES BY STAGE:\n")
    for stage in ['original', 'backtranslate', 'fasttext', 'ulmfit']:
        stage_df = df[df['stage'] == stage]
        if len(stage_df) > 0:
            print(f"--- {stage.upper()} ---")
            for _, row in stage_df.head(3).iterrows():
                print(f"[{row['intent']}] {row['text']}")
            print()
    
    print("="*80 + "\n")
    
    return df


# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Generate dataset with full pipeline
    df = generate_sequential_augmented_dataset(
        target_total=30000,
        augment_prob=0.3,        # 50% ของ samples จะถูก augment
        max_variants=3,          # สูงสุด 3 variants ต่อ original
        use_backtranslate=True,  # เปิดใช้ backtranslate
        use_fasttext=True,       # เปิดใช้ fastText
        use_ulmfit=True,         # เปิดใช้ ULMFit
        use_gpu=False,           # True ถ้ามี GPU
        save_path="../Train_model_IntentData/intents_dataset_sequential.csv"
    )
    
    print("🎉 Dataset generation complete!")
    print(f"📈 Final dataset size: {len(df):,} samples")
    print(f"📊 Augmented samples: {len(df[df['stage'] != 'original']):,}")