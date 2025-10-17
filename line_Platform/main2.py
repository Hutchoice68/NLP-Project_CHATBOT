import os
import time
import subprocess
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
    TextMessage
)

from bb import reponse_message


app = FastAPI()

# โหลด environment variables
load_dotenv(override=True)

get_access_token = os.getenv('ACCESS_TOKEN')
get_channel_secret = os.getenv('CHANNEL_SECRET')

if not get_access_token or not get_channel_secret:
    raise ValueError("กรุณาตั้งค่า ACCESS_TOKEN และ CHANNEL_SECRET ในไฟล์ .env")

configuration = Configuration(access_token=get_access_token)
handler = WebhookHandler(channel_secret=get_channel_secret)


@app.get("/")
async def root():
    return {"message": "LINE Bot is running!", "status": "ok"}


@app.post("/callback")
async def callback(request: Request, x_line_signature: str = Header(None)):
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
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            reply_message = reponse_message(event)
            if not reply_message:
                return None
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[reply_message]
                )
            )
    except Exception as e:
        print(f"Error handling message: {e}")
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


def ensure_ollama_running():
    """ตรวจสอบว่า Ollama server ทำงานอยู่หรือไม่ ถ้าไม่ให้เริ่มอัตโนมัติ"""
    try:
        result = subprocess.run(["pgrep", "-f", "ollama"], stdout=subprocess.PIPE)
        if result.returncode == 0:
            print("✅ Ollama already running.")
            return

        print("🚀 Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("✅ Ollama started successfully.")
    except Exception as e:
        print(f"⚠️ Unable to start Ollama automatically: {e}")


if __name__ == "__main__":
    ensure_ollama_running()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
