# Intent project with line and web platform
## ขั้นตอนที่ 1: ติดตั้ง Dependencies

```
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

## ขั้นตอนที่ 2: ตั้งค่า Environment Variables
```
# 1.สร้างไฟล์ .env

# 2. ข้อมูลใน .env
# --- LINE API ---
ACCESS_TOKEN=
CHANNEL_SECRET=

# --- URL ---
AI_CORE_URL=http://localhost:8001 #(แล้วแต่ว่าจะตั้งที่ port ไหน)

# --- Path intent model ---
MODEL_PATH=../models/thai_intent_model.pkl

# --- LLM model ---
OLLAMA_MODEL=llama3.2
```

## ขั้นตอนที่ 3: รัน Ollama (ถ้าต้องการใช้ LLM)
Terminal 0 - เปิดก่อนทุกอย่าง
```
# รัน Ollama
ollama serve

# ใน terminal อื่น - ดาวน์โหลดโมเดล (ครั้งแรกเท่านั้น)
ollama pull llama3.2
```

## ลำดับการรัน Services
### รัน AI Core ก่อน เพื่อเปิดเซิฟเวอร์หลัก
(สำคัญ)
```
#ต้องเข้าโฟลเดอร์ main ถึงจะรันเปิดเซิฟได้
cd main
```
Terminal 1
```
uvicorn ai_core:app --host 0.0.0.0 --port 8001 --reload
```

### รัน LINE Bot
Terminal 2 
```
uvicorn line:app --host 0.0.0.0 --port 8000 --reload
```
Terminal 3 อันนี้ใช้เปิด webhook ผ่าน ngrok(ต้องไปติดตั้งลงเครื่องก่อน)
```
ngrok http 8000
```
### รัน Web
Terminal 4 
```
uvicorn web:app --host 0.0.0.0 --port 8002 --reload
```
จากนั้นก็เปิดบราวเซอร์: http://localhost:8002
