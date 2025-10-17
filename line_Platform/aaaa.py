# ติดตั้งก่อนใช้งาน: pip install pythainlp

import os
import pickle
import requests
import numpy as np
from linebot.v3.messaging import TextMessage
from datetime import datetime
from dotenv import load_dotenv

# ✨ ใช้ PyThaiNLP สำหรับแยกคำภาษาไทยที่แม่นยำ
try:
    from pythainlp.tokenize import word_tokenize
    USE_PYTHAINLP = True
    print("✅ PyThaiNLP loaded successfully")
except ImportError:
    USE_PYTHAINLP = False
    print("⚠️ PyThaiNLP not found - using simple tokenizer")
    import re

# โหลด .env
load_dotenv(override=True)

MODEL_PATH = os.getenv('MODEL_PATH', 'thai_intent_model.pkl')

# ===== MOCK DATABASE =====
MOCK_PRODUCTS = [
    {
        "product_id": 1,
        "product_name": "เสื้อยืดสีขาว",
        "description": "เสื้อยืดคอกลมผ้าคอตตอน 100% สีขาว ใส่สบาย",
        "price": 299,
        "stock": 50,
        "keywords": ["เสื้อ", "เสื้อยืด", "สีขาว", "ผ้า", "คอตตอน"],
    },
    {
        "product_id": 2,
        "product_name": "กางเกงยีนส์",
        "description": "กางเกงยีนส์ขายาว ทรงสลิม สีน้ำเงินเข้ม",
        "price": 890,
        "stock": 30,
        "keywords": ["กางเกง", "ยีนส์", "ขายาว", "สลิม", "น้ำเงิน"],
    },
    {
        "product_id": 3,
        "product_name": "รองเท้าผ้าใบ",
        "description": "รองเท้าผ้าใบสีดำ น้ำหนักเบา ใส่สบาย",
        "price": 1290,
        "stock": 20,
        "keywords": ["รองเท้า", "ผ้าใบ", "สีดำ", "เบา"],
    },
    {
        "product_id": 4,
        "product_name": "กระเป๋าสะพายข้าง",
        "description": "กระเป๋าผ้าแคนวาส ขนาดกลาง สะพายข้างสบาย",
        "price": 450,
        "stock": 15,
        "keywords": ["กระเป๋า", "สะพาย", "แคนวาส"],
    },
    {
        "product_id": 5,
        "product_name": "หมวกแก๊ป",
        "description": "หมวกแก๊ปปรับขนาดได้ ผ้าคอตตอน",
        "price": 250,
        "stock": 40,
        "keywords": ["หมวก", "แก๊ป", "คอตตอน"],
    }
]

MOCK_ORDERS = {
    "U1234567890": [
        {
            "order_id": "ORD001",
            "product_name": "เสื้อยืดสีขาว",
            "status": "จัดส่งแล้ว",
            "created_at": "2025-10-15 10:30:00",
        }
    ],
    "default": [
        {
            "order_id": "ORD999",
            "product_name": "สินค้าตัวอย่าง",
            "status": "รอชำระเงิน",
            "created_at": "2025-10-17 09:00:00",
        }
    ]
}

# ===== LOAD MODEL =====
intent_model = None
model_components = None

try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_data = pickle.load(f)
    
    if isinstance(loaded_data, dict):
        model_components = loaded_data
        intent_model = 'hybrid'
        print(f"✅ Loaded hybrid model from: {MODEL_PATH}")
    else:
        intent_model = loaded_data
        print(f"✅ Loaded intent model from: {MODEL_PATH}")
except:
    print(f"⚠️ Model not found - using rule-based")


# ==========================
# Thai Text Processing
# ==========================
def thai_tokenize(text):
    """✨ แยกคำภาษาไทย - ใช้ PyThaiNLP ถ้ามี, ไม่เช่นนั้นใช้ fallback"""
    if USE_PYTHAINLP:
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
        tokens = [t for t in tokens if len(t.strip()) >= 2 and not t.isspace()]
        return tokens
    else:
        THAI_DICT = [
            'ต้องการ', 'รายละเอียด', 'ข้อมูล', 'กางเกง', 'เสื้อ', 'รองเท้า',
            'กระเป๋า', 'หมวก', 'ราคา', 'สต็อก', 'สั่งซื้อ', 'คืนสินค้า',
            'ยีนส์', 'ผ้าใบ', 'สีขาว', 'สีดำ', 'คอตตอน', 'ขอ', 'ดู', 'มี'
        ]
        THAI_DICT.sort(key=len, reverse=True)
        
        text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text.lower())
        found = []
        i = 0
        while i < len(text):
            matched = False
            for word in THAI_DICT:
                if text[i:i+len(word)] == word:
                    found.append(word)
                    i += len(word)
                    matched = True
                    break
            if not matched:
                i += 1
        return list(set(found))


# ==========================
# Database Functions
# ==========================
def get_product_info(product_name=None, limit=5):
    """ดึงข้อมูลสินค้า"""
    print(f"🔍 get_product_info: '{product_name}'")
    
    if product_name:
        search_terms = thai_tokenize(product_name)
        print(f"   Tokens: {search_terms}")
        
        if not search_terms:
            search_terms = [product_name.lower()]
        
        results = []
        for p in MOCK_PRODUCTS:
            searchable = (
                f"{p['product_name']} {p['description']} "
                f"{' '.join(p.get('keywords', []))}"
            ).lower()
            
            match_count = sum(1 for t in search_terms if t in searchable)
            if match_count > 0:
                results.append((p, match_count))
                print(f"   ✅ {p['product_name']} (score: {match_count})")
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in results[:limit]]
    
    return MOCK_PRODUCTS[:limit]


def get_order_info(user_id=None, order_id=None):
    """ดึงข้อมูลคำสั่งซื้อ"""
    if order_id:
        for orders in MOCK_ORDERS.values():
            for o in orders:
                if o['order_id'] == order_id:
                    return [o]
        return []
    elif user_id:
        return MOCK_ORDERS.get(user_id, MOCK_ORDERS.get("default", []))[:5]
    return []


# ==========================
# LLM (Ollama Local)
# ==========================
def call_ollama_llm(prompt, context=""):
    """เรียก LLM จาก Ollama ในเครื่อง"""
    try:
        print(f"🔄 Calling Ollama...")

        # 🔧 ตั้งชื่อโมเดลที่คุณมี เช่น "llama3.2" หรือ "qwen2.5"
        model_name = "llama3.2"

        system_prompt = f"คุณคือผู้ช่วยร้านค้า ตอบเป็นภาษาไทยสั้นๆ 2-3 ประโยค\n\nบริบท:\n{context}\n\nผู้ใช้: {prompt}"

        payload = {
            "model": model_name,
            "prompt": system_prompt,
            "stream": False
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "").strip()
            print(f"   ✅ Ollama Response OK")
            return answer
        else:
            print(f"   ❌ Ollama Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Ollama Error: {e}")
        return None


# ==========================
# Intent Prediction
# ==========================
def predict_intent(text):
    """ทำนาย Intent"""
    global intent_model, model_components
    
    if intent_model == 'hybrid' and model_components:
        try:
            semantic_model = model_components['semantic']
            tfidf_vectorizer = model_components['tfidf']
            clf = model_components['clf']
            
            semantic_features = semantic_model.encode([text])
            tfidf_features = tfidf_vectorizer.transform([text]).toarray()
            
            scaler_sem = model_components.get('scaler_sem')
            scaler_tfidf = model_components.get('scaler_tfidf')
            
            if scaler_sem:
                semantic_features = scaler_sem.transform(semantic_features)
            if scaler_tfidf:
                tfidf_features = scaler_tfidf.transform(tfidf_features)
            
            alpha = model_components.get('alpha', 0.5)
            beta = model_components.get('beta', 0.5)
            
            combined = np.hstack([alpha * semantic_features, beta * tfidf_features])
            intent = clf.predict(combined)[0]
            print(f"🎯 Intent: {intent}")
            return intent
        except Exception as e:
            print(f"⚠️ Model error: {e}")
    
    # Rule-based fallback
    text_lower = text.lower()
    if any(w in text_lower for w in ['สวัสดี', 'hello', 'hi']):
        return "greeting"
    elif any(w in text_lower for w in ['ขอบคุณ', 'thank']):
        return "thank_you"
    elif any(w in text_lower for w in ['สั่ง', 'ซื้อ', 'order']):
        return "order_product"
    elif any(w in text_lower for w in ['คืน', 'refund', 'เคลม']):
        return "refund_request"
    elif any(w in text_lower for w in ['ช่วย', 'help', 'สอบถาม']):
        return "help_request"
    elif any(w in text_lower for w in ['ราคา', 'สินค้า', 'เสื้อ', 'กางเกง', 'รองเท้า']):
        return "ask_info"
    elif any(w in text_lower for w in ['รีวิว', 'feedback']):
        return "feedback"
    else:
        return "unknown"


# ==========================
# Intent Response
# ==========================
def handle_intent_response(intent, user_message, user_id):
    """จัดการตอบคำถาม"""
    if intent == "greeting":
        return "สวัสดีครับ! ยินดีต้อนรับค่ะ 😊"
    elif intent == "thank_you":
        return "ยินดีครับ! มีอะไรให้ช่วยเพิ่มเติมไหมคะ 🙏"
    elif intent == "ask_info":
        products = get_product_info(user_message, limit=3)
        if products:
            context = "\n".join([
                f"- {p['product_name']}: {p['price']} บาท, {p['description']}, คงเหลือ {p['stock']} ชิ้น"
                for p in products
            ])
            llm_response = call_ollama_llm(user_message, f"สินค้า:\n{context}")
            if llm_response:
                return llm_response
            product_list = "\n".join([
                f"• {p['product_name']}\n  💰 {p['price']} บาท\n  📦 คงเหลือ {p['stock']} ชิ้น"
                for p in products
            ])
            return f"✨ พบสินค้าที่คุณสนใจ:\n\n{product_list}\n\nสนใจสั่งซื้อไหมคะ?"
        else:
            return "ขออภัยครับ ไม่พบสินค้าที่คุณค้นหา 😅\n\nลองค้นหา: เสื้อ, กางเกง, รองเท้า, กระเป๋า, หมวก"
    elif intent == "order_product":
        orders = get_order_info(user_id=user_id)
        if orders:
            order_list = "\n".join([f"• {o['order_id']}: {o['status']}" for o in orders[:2]])
            return f"คำสั่งซื้อของคุณ:\n{order_list}"
        return "ต้องการสั่งซื้อสินค้าไหมคะ? บอกชื่อสินค้าได้เลยค่ะ"
    elif intent == "refund_request":
        return "สามารถคืนสินค้าได้ภายใน 7 วัน กรุณาแจ้งหมายเลขคำสั่งซื้อค่ะ 📦"
    elif intent == "help_request":
        return "ยินดีช่วยเหลือครับ! สามารถสอบถามเรื่อง:\n• สินค้า\n• คำสั่งซื้อ\n• การคืนสินค้า"
    elif intent == "feedback":
        return "ขอบคุณสำหรับ Feedback ครับ! เราจะนำไปปรับปรุงค่ะ 💪"
    else:
        return "ขออภัยครับ ไม่เข้าใจคำถาม ลองถามใหม่ได้ไหมคะ? 🤔"


# ==========================
# Main Response
# ==========================
def reponse_message(event):
    """ฟังก์ชันหลัก"""
    user_message = event.message.text.strip()
    user_id = getattr(event.source, 'user_id', 'default')
    
    print(f"\n{'='*50}")
    print(f"📨 Message: {user_message}")
    print(f"👤 User: {user_id}")
    
    intent = predict_intent(user_message)
    response = handle_intent_response(intent, user_message, user_id)
    
    print(f"💬 Response: {response}")
    print(f"{'='*50}\n")
    
    return TextMessage(text=response)
