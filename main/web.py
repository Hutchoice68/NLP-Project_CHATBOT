"""
Web Chat Application - เชื่อมต่อกับ AI Core Service
รัน: uvicorn web_chat:app --host 0.0.0.0 --port 8002 --reload
เปิดใน browser: http://localhost:8002
"""
import os
import requests
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="Web Chat Application", version="1.0.0")

AI_CORE_URL = os.getenv('AI_CORE_URL', 'http://localhost:8001')


class ChatMessage(BaseModel):
    message: str
    session_id: str = None


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    """หน้า Chat UI"""
    return """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chat - Web Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 600px;
            height: 700px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #messageInput:focus {
            border-color: #667eea;
        }
        
        #sendButton {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        
        #sendButton:hover {
            transform: scale(1.05);
        }
        
        #sendButton:active {
            transform: scale(0.95);
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: white;
            border-radius: 18px;
            width: fit-content;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background: #667eea;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes bounce {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }
        
        .status {
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            🤖 AI Chat Assistant
        </div>
        <div class="status" id="status">เชื่อมต่อกับ AI Core...</div>
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    
                    
                </div>
            </div>
        </div>
        <div class="typing-indicator" id="typingIndicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <div class="chat-input-container">
            <input 
                type="text" 
                id="messageInput" 
                placeholder="พิมพ์ข้อความ..." 
                autocomplete="off"
            />
            <button id="sendButton">ส่ง</button>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const statusDiv = document.getElementById('status');
        
        // สร้าง session ID สำหรับ user
        const sessionId = 'web_' + Math.random().toString(36).substr(2, 9);
        
        // ตรวจสอบ AI Core status
        checkAIStatus();
        
        async function checkAIStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                if (data.ai_core_status === 'connected') {
                    statusDiv.textContent = '✅ เชื่อมต่อ AI Core สำเร็จ';
                    statusDiv.style.color = '#4caf50';
                } else {
                    statusDiv.textContent = '⚠️ ไม่สามารถเชื่อมต่อ AI Core';
                    statusDiv.style.color = '#f44336';
                }
                setTimeout(() => statusDiv.style.display = 'none', 3000);
            } catch (error) {
                statusDiv.textContent = '❌ เกิดข้อผิดพลาดในการเชื่อมต่อ';
                statusDiv.style.color = '#f44336';
            }
        }
        
        function addMessage(text, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = text;
            
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function showTyping() {
            typingIndicator.style.display = 'block';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function hideTyping() {
            typingIndicator.style.display = 'none';
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // แสดงข้อความของ user
            addMessage(message, true);
            messageInput.value = '';
            
            // แสดง typing indicator
            showTyping();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                // ซ่อน typing indicator
                hideTyping();
                
                // แสดงคำตอบจาก bot
                addMessage(data.response, false);
                
            } catch (error) {
                hideTyping();
                addMessage('ขออภัยครับ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ', false);
                console.error('Error:', error);
            }
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Focus ที่ input ตอนโหลดหน้า
        messageInput.focus();
    </script>
</body>
</html>
    """


@app.post("/api/chat")
async def chat(message: ChatMessage):
    """API endpoint สำหรับ Web Chat"""
    try:
        # สร้าง user_id จาก session_id
        user_id = message.session_id if message.session_id else f"web_{uuid.uuid4().hex[:8]}"
        
        # เรียก AI Core Service
        response = requests.post(
            f"{AI_CORE_URL}/chat",
            json={
                "message": message.message,
                "user_id": user_id,
                "platform": "web"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "response": "ขออภัยครับ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งค่ะ",
                "intent": "error"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "response": "ไม่สามารถเชื่อมต่อระบบ AI ได้ กรุณาตรวจสอบว่า AI Core Service ทำงานอยู่",
            "intent": "error"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "response": "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ",
            "intent": "error"
        }


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    try:
        ai_response = requests.get(f"{AI_CORE_URL}/health", timeout=5)
        ai_status = "connected" if ai_response.status_code == 200 else "disconnected"
    except:
        ai_status = "disconnected"
    
    return {
        "status": "running",
        "ai_core_status": ai_status,
        "ai_core_url": AI_CORE_URL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)