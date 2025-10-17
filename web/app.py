import streamlit as st
import json
import random
from utils.predict_intent import predict_intent
from langchain_community.chat_models import ChatOllama

# ===================== MOCK DATABASE =====================
MOCK_PRODUCTS = [
    {"product_id": 1, "product_name": "เสื้อยืดสีขาว", "description": "เสื้อยืดคอกลมผ้าคอตตอน 100% สีขาว ใส่สบาย", "price": 299, "stock": 50, "keywords": ["เสื้อ", "เสื้อยืด", "สีขาว", "ผ้า", "คอตตอน"]},
    {"product_id": 2, "product_name": "กางเกงยีนส์", "description": "กางเกงยีนส์ขายาว ทรงสลิม สีน้ำเงินเข้ม", "price": 890, "stock": 30, "keywords": ["กางเกง", "ยีนส์", "ขายาว", "สลิม", "น้ำเงิน"]},
    {"product_id": 3, "product_name": "รองเท้าผ้าใบ", "description": "รองเท้าผ้าใบสีดำ น้ำหนักเบา ใส่สบาย", "price": 1290, "stock": 20, "keywords": ["รองเท้า", "ผ้าใบ", "สีดำ", "เบา"]},
    {"product_id": 4, "product_name": "กระเป๋าสะพายข้าง", "description": "กระเป๋าผ้าแคนวาส ขนาดกลาง สะพายข้างสบาย", "price": 450, "stock": 15, "keywords": ["กระเป๋า", "สะพาย", "แคนวาส"]},
    {"product_id": 5, "product_name": "หมวกแก๊ป", "description": "หมวกแก๊ปปรับขนาดได้ ผ้าคอตตอน", "price": 250, "stock": 40, "keywords": ["หมวก", "แก๊ป", "คอตตอน"]},
]

MOCK_ORDERS = {
    "U1234567890": [
        {"order_id": "ORD001", "product_name": "เสื้อยืดสีขาว", "status": "จัดส่งแล้ว", "created_at": "2025-10-15 10:30:00"},
    ],
    "default": [
        {"order_id": "ORD999", "product_name": "สินค้าตัวอย่าง", "status": "รอชำระเงิน", "created_at": "2025-10-17 09:00:00"},
    ],
}

# ===================== ตั้งค่า Ollama =====================
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Thai Chatbot E-Commerce", page_icon="🛍️", layout="centered")
st.title("🛍️ Thai E-Commerce Chatbot")
st.caption("ระบบผู้ช่วยขายสินค้าออนไลน์ (Intent + Ollama + Streamlit)")

# โหลดข้อความตอบกลับ intent
with open("intent_responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

# เก็บประวัติการแชท
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===================== ฟังก์ชัน =====================
def get_response(intent):
    if intent in responses:
        return random.choice(responses[intent])
    return random.choice(responses["unknown"])

def search_product(text):
    result = [p for p in MOCK_PRODUCTS if any(k in text for k in p["keywords"])]
    return result

def get_order(user_id="default"):
    return MOCK_ORDERS.get(user_id, MOCK_ORDERS["default"])

# ===================== Input จากผู้ใช้ =====================
user_input = st.chat_input("พิมพ์ข้อความที่นี่...")

if user_input:
    # วิเคราะห์ intent
    intent = predict_intent(user_input)
    base_reply = get_response(intent)
    ai_reply = ""

    # เชื่อม intent กับ mock database
    if intent == "search_product":
        products = search_product(user_input)
        if products:
            product_text = "\n".join([f"- {p['product_name']} ({p['price']} บาท)" for p in products])
            ai_reply = f"{base_reply}\n\nพบสินค้าที่ใกล้เคียง:\n{product_text}"
        else:
            ai_reply = f"{base_reply}\n\nไม่พบสินค้าที่เกี่ยวข้องค่ะ 🛒"

    elif intent == "check_order":
        orders = get_order("U1234567890")
        order_text = "\n".join([f"- หมายเลขคำสั่งซื้อ {o['order_id']} : {o['product_name']} ({o['status']})" for o in orders])
        ai_reply = f"{base_reply}\n\n{order_text}"

    else:
        # ใช้ LLM Ollama ช่วยตอบ
        llm_response = llm.invoke(user_input)
        ai_reply = f"{base_reply}\n\n💬 {llm_response.content}"

    # บันทึกลง session
    st.session_state.chat_history.append(("🧑‍💬", user_input))
    st.session_state.chat_history.append(("🤖", f"[{intent}] {ai_reply}"))

# ===================== แสดงประวัติแชท =====================
for sender, msg in st.session_state.chat_history:
    bg_color = "#DCF8C6" if sender == "🧑‍💬" else "#4A90E2"
    text_color = "#000000" if sender == "🧑‍💬" else "#FFFFFF"
    st.markdown(
        f"<div style='background-color:{bg_color};color:{text_color};padding:10px;border-radius:8px;margin-bottom:5px;'>"
        f"<strong>{sender}</strong>: {msg}</div>",
        unsafe_allow_html=True,
    )
