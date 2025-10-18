"""
LINE Bot - เชื่อมต่อกับ AI Core Service
รัน: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Header

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from linebot.v3.messaging import (
    ApiClient, 
    MessagingApi, 
    Configuration, 
    ReplyMessageRequest, 
    TextMessage,
)

# โหลด environment variables
load_dotenv(override=True)

app = FastAPI(title="LINE Bot Gateway", version="1.0.0")

# LINE Configuration
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')
AI_CORE_URL = os.getenv('AI_CORE_URL', 'http://localhost:8001')

if not ACCESS_TOKEN or not CHANNEL_SECRET:
    raise ValueError("กรุณาตั้งค่า ACCESS_TOKEN และ CHANNEL_SECRET ในไฟล์ .env")

configuration = Configuration(access_token=ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)


def call_ai_core(message: str, user_id: str) -> dict:
    """
    เรียก AI Core Service แทนการโหลดโมเดลเอง
    """
    try:
        response = requests.post(
            f"{AI_CORE_URL}/chat",
            json={
                "message": message,
                "user_id": user_id,
                "platform": "line"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ AI Core Error: {response.status_code}")
            return {
                "response": "ขออภัยครับ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ",
                "intent": "error"
            }
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to AI Core Service")
        return {
            "response": "ไม่สามารถเชื่อมต่อระบบ AI ได้ กรุณาตรวจสอบว่า AI Core Service ทำงานอยู่",
            "intent": "error"
        }
    except Exception as e:
        print(f"❌ Error calling AI Core: {e}")
        return {
            "response": "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ",
            "intent": "error"
        }


@app.get("/")
async def root():
    """Health check endpoint"""
    # ตรวจสอบการเชื่อมต่อกับ AI Core
    try:
        ai_response = requests.get(f"{AI_CORE_URL}/health", timeout=5)
        ai_status = "connected" if ai_response.status_code == 200 else "disconnected"
    except:
        ai_status = "disconnected"
    
    return {
        "service": "LINE Bot Gateway",
        "status": "running",
        "ai_core_status": ai_status,
        "ai_core_url": AI_CORE_URL
    }


@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(None)):
    """Webhook callback สำหรับรับข้อความจาก LINE"""
    body = await request.body()
    body_str = body.decode('utf-8')
    
    try:
        handler.handle(body_str, x_line_signature)
    except InvalidSignatureError:
        print("❌ Invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    """จัดการข้อความที่ได้รับจากผู้ใช้"""
    user_message = event.message.text.strip()
    user_id = getattr(event.source, 'user_id', 'default')
    
    print(f"\n{'='*50}")
    print(f"📱 LINE Message from {user_id}")
    print(f"💬 Message: {user_message}")
    
    try:
        # เรียก AI Core แทนการประมวลผลเอง
        ai_response = call_ai_core(user_message, user_id)
        response_text = ai_response.get('response', 'ขออภัยครับ ไม่สามารถประมวลผลได้')
        
        print(f"✅ AI Intent: {ai_response.get('intent', 'unknown')}")
        print(f"✅ Response: {response_text[:100]}...")
        print(f"{'='*50}\n")
        
        # ส่งข้อความกลับไปที่ LINE
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=response_text)]
                )
            )
            
    except Exception as e:
        print(f"❌ Error handling message: {e}")
        try:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                error_message = TextMessage(
                    text="ขออภัยครับ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ"
                )
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[error_message]
                    )
                )
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)