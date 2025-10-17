import os
import pickle
import requests
import mysql.connector
import numpy as np
from linebot.v3.messaging import TextMessage

# โหลด environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'your_database')

# โหลดโมเดล Intent Classification
MODEL_PATH = os.getenv('MODEL_PATH', 'thai_intent_model.pkl')

# ===== GLOBAL VARIABLES =====
intent_model = None
model_components = None

try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_data = pickle.load(f)
    
    if isinstance(loaded_data, dict):
        print(f"✅ โหลด dict จาก: {MODEL_PATH}")
        print(f"   Keys ที่มี: {loaded_data.keys()}")
        
        # ✅ ตั้งค่าสำหรับ hybrid model
        model_components = loaded_data
        intent_model = 'hybrid'
        print("   → ใช้ Hybrid Model (semantic + tfidf + clf)")
        
    else:
        intent_model = loaded_data
        model_components = None
        print(f"✅ โหลดโมเดล Intent สำเร็จจาก: {MODEL_PATH}")
        
except FileNotFoundError:
    print(f"⚠️ ไม่พบไฟล์โมเดล: {MODEL_PATH}")
    print("   → ใช้ Rule-based intent classification")
    intent_model = None
    model_components = None
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
    intent_model = None
    model_components = None


# ==========================
# Database Connection
# ==========================
def get_db_connection():
    """สร้างการเชื่อมต่อกับ MySQL Database"""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4'
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None


def get_product_info(product_name=None, limit=5):
    """ดึงข้อมูลสินค้าจาก database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        if product_name:
            query = """
                SELECT * FROM products 
                WHERE product_name LIKE %s OR description LIKE %s 
                LIMIT %s
            """
            cursor.execute(query, (f'%{product_name}%', f'%{product_name}%', limit))
        else:
            query = "SELECT * FROM products LIMIT %s"
            cursor.execute(query, (limit,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    
    except mysql.connector.Error as err:
        print(f"Query error: {err}")
        return []


def get_order_info(user_id=None, order_id=None):
    """ดึงข้อมูลคำสั่งซื้อจาก database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        if order_id:
            query = "SELECT * FROM orders WHERE order_id = %s"
            cursor.execute(query, (order_id,))
        elif user_id:
            query = "SELECT * FROM orders WHERE user_id = %s ORDER BY created_at DESC LIMIT 5"
            cursor.execute(query, (user_id,))
        else:
            return []
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    
    except mysql.connector.Error as err:
        print(f"Query error: {err}")
        return []


# ==========================
# OpenRouter API
# ==========================
def call_openrouter_llm(prompt, context=""):
    """เรียกใช้ OpenRouter LLM API"""
    
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == 'your_openrouter_api_key':
        print("⚠️ ไม่มี OPENROUTER_API_KEY - ใช้คำตอบสำเร็จรูป")
        return None
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-exp:free",
                "messages": [
                    {
                        "role": "system",
                        "content": f"คุณคือผู้ช่วยลูกค้าที่เป็นมิตร ตอบคำถามเป็นภาษาไทย สั้นและกระชับ ไม่เกิน 3-4 ประโยค\n\nข้อมูลบริบท:\n{context}"
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"OpenRouter API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")
        return None


# ==========================
# Intent Prediction
# ==========================
def predict_intent(text):
    """ทำนาย Intent จากข้อความ"""
    global intent_model, model_components

    # Hybrid Model
    if intent_model == 'hybrid' and model_components:
        try:
            print("🔍 ใช้ Hybrid Model ทำการทำนาย intent ...")

            # 1. Semantic Features
            semantic_model = model_components['semantic']
            semantic_features = semantic_model.encode([text])

            # 2. TF-IDF Features
            tfidf_vectorizer = model_components['tfidf']
            tfidf_features = tfidf_vectorizer.transform([text]).toarray()

            # 3. Scaling
            scaler_sem = model_components['scaler_sem']
            scaler_tfidf = model_components['scaler_tfidf']

            if scaler_sem:
                semantic_features = scaler_sem.transform(semantic_features)
            if scaler_tfidf:
                tfidf_features = scaler_tfidf.transform(tfidf_features)

            # 4. Combine with weight
            alpha = model_components.get('alpha', 0.5)
            beta = model_components.get('beta', 0.5)

            combined_features = np.hstack([
                alpha * semantic_features,
                beta * tfidf_features
            ])

            # 5. Predict
            clf = model_components['clf']
            intent = clf.predict(combined_features)[0]
            print(f"🎯 Predicted intent: {intent}")
            return intent

        except Exception as e:
            print(f"⚠️ Hybrid model prediction error: {e}")
            print("→ Fallback to Rule-based")

    # Rule-based fallback
    print("⚠️ ใช้ Rule-based fallback")
    text_lower = text.lower()

    if any(word in text_lower for word in ['สวัสดี', 'hello', 'hi', 'ดีครับ', 'ดีค่ะ']):
        return "greeting"
    elif any(word in text_lower for word in ['ขอบคุณ', 'thank']):
        return "thank_you"
    elif any(word in text_lower for word in ['สั่ง', 'ซื้อ', 'order', 'จอง']):
        return "order_product"
    elif any(word in text_lower for word in ['คืน', 'refund', 'เปลี่ยน']):
        return "refund_request"
    elif any(word in text_lower for word in ['ช่วย', 'help', 'ติดต่อ', 'สอบถาม']):
        return "help_request"
    elif any(word in text_lower for word in ['ราคา', 'สินค้า', 'สต็อก', 'price']):
        return "ask_info"
    elif any(word in text_lower for word in ['รีวิว', 'feedback', 'แนะนำ']):
        return "feedback"
    else:
        return "unknown"


# ==========================
# Intent Response
# ==========================
def handle_intent_response(intent, user_message, user_id):
    """จัดการตอบคำถามตาม Intent ที่ทำนายได้"""
    
    if intent == "greeting":
        return "สวัสดีครับ! ยินดีต้อนรับค่ะ 😊 มีอะไรให้ช่วยไหมคะ?"
    
    elif intent == "thank_you":
        return "ยินดีครับ! หากมีคำถามเพิ่มเติม สอบถามได้ตลอดเลยนะคะ 🙏"
    
    elif intent == "ask_info":
        products = get_product_info(user_message, limit=3)
        if products:
            product_context = "\n".join([
                f"- {p.get('product_name', 'N/A')}: ราคา {p.get('price', 'N/A')} บาท, {p.get('description', '')}"
                for p in products
            ])
            context = f"ข้อมูลสินค้าในระบบ:\n{product_context}"
        else:
            context = "ไม่พบข้อมูลสินค้าในระบบ"
        
        llm_response = call_openrouter_llm(user_message, context)
        return llm_response or "ขออภัยครับ ไม่สามารถตอบคำถามได้ในขณะนี้"
    
    elif intent == "order_product":
        orders = get_order_info(user_id=user_id)
        if orders:
            order_context = "\n".join([
                f"- คำสั่งซื้อ #{o.get('order_id')}: สถานะ {o.get('status', 'N/A')}, วันที่ {o.get('created_at', 'N/A')}"
                for o in orders[:3]
            ])
            context = f"ประวัติคำสั่งซื้อ:\n{order_context}"
        else:
            context = "ไม่พบประวัติการสั่งซื้อ"
        
        llm_response = call_openrouter_llm(user_message, context)
        return llm_response or "หากต้องการสั่งซื้อสินค้า กรุณาแจ้งชื่อสินค้าที่ต้องการค่ะ"
    
    elif intent == "refund_request":
        context = "นโยบายการคืนสินค้า: สามารถคืนสินค้าได้ภายใน 7 วัน หากสินค้ามีปัญหาหรือไม่ตรงตามที่สั่ง"
        llm_response = call_openrouter_llm(user_message, context)
        return llm_response or "หากต้องการขอคืนสินค้า กรุณาแจ้งหมายเลขคำสั่งซื้อค่ะ"
    
    elif intent == "help_request":
        llm_response = call_openrouter_llm(
            user_message,
            "คุณสามารถช่วยเหลือลูกค้าในเรื่อง: สอบถามสินค้า, ตรวจสอบคำสั่งซื้อ, ขอคืนสินค้า"
        )
        return llm_response or "สามารถสอบถามข้อมูลสินค้า หรือติดตามคำสั่งซื้อได้เลยครับ!"
    
    elif intent == "feedback":
        context = "ขอบคุณสำหรับความคิดเห็น เราจะนำไปพัฒนาปรับปรุงให้ดีขึ้นค่ะ"
        llm_response = call_openrouter_llm(user_message, context)
        return llm_response or "ขอบคุณสำหรับ Feedback ครับ! 💪"
    
    else:
        llm_response = call_openrouter_llm(user_message, "ตอบคำถามทั่วไปเกี่ยวกับการช้อปปิ้งและสินค้า")
        return llm_response or "ขออภัยครับ ไม่เข้าใจคำถาม สามารถอธิบายเพิ่มเติมได้ไหมคะ?"


# ==========================
# LINE Message Response
# ==========================
def reponse_message(event):
    """ฟังก์ชันหลักสำหรับตอบข้อความ"""
    user_message = event.message.text.strip()
    user_id = getattr(event.source, 'user_id', None)
    
    intent = predict_intent(user_message)
    print(f"User message: {user_message}")
    print(f"Predicted intent: {intent}")
    
    response_text = handle_intent_response(intent, user_message, user_id)
    
    return TextMessage(text=response_text)
