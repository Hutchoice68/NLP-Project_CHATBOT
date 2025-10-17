import os
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
    # FlexMessage, 
    # Emoji,
)

# เปลี่ยนจาก response_message เป็น reponse_message (ตามชื่อไฟล์เดิมของคุณ)
from aaaa import reponse_message


app = FastAPI()

# โหลด environment variables
load_dotenv(override=True)

# LINE Access Key
get_access_token = os.getenv('ACCESS_TOKEN')
# LINE Secret Key
get_channel_secret = os.getenv('CHANNEL_SECRET')

# ตรวจสอบว่ามี token และ secret หรือไม่
if not get_access_token or not get_channel_secret:
    raise ValueError("กรุณาตั้งค่า ACCESS_TOKEN และ CHANNEL_SECRET ในไฟล์ .env")

configuration = Configuration(access_token=get_access_token)
handler = WebhookHandler(channel_secret=get_channel_secret)


@app.get("/")
async def root():
    """Endpoint สำหรับตรวจสอบว่า server ทำงานหรือไม่"""
    return {"message": "LINE Bot is running!", "status": "ok"}


@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(None)):
    """Webhook callback สำหรับรับข้อความจาก LINE"""
    body = await request.body()
    body_str = body.decode('utf-8')
    
    try:
        handler.handle(body_str, x_line_signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature.")

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    """จัดการข้อความที่ได้รับจากผู้ใช้"""
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)

            # เรียกใช้ฟังก์ชัน reponse_message เพื่อสร้างคำตอบ
            reply_message = reponse_message(event)

            # ถ้าไม่มีการตอบกลับ ให้ข้ามไป
            if not reply_message:
                return None

            # ส่งข้อความตอบกลับ
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[reply_message]
                )
            )
    except Exception as e:
        print(f"Error handling message: {e}")
        # สามารถส่งข้อความ error กลับไปหาผู้ใช้ได้ (optional)
        try:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                error_message = TextMessage(text="ขออภัยครับ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ")
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[error_message]
                    )
                )
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")