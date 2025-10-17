import random
import json
import subprocess

# โหลด intent responses จากไฟล์ JSON
with open("intent_responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)


def generate_response(intent, user_input, context):
    """
    ฟังก์ชันตอบกลับข้อความ
    - ถ้ามี intent ใน responses.json → ตอบจาก preset
    - ถ้าไม่รู้จัก intent → เรียกโมเดล Ollama (Llama 3.2)
    """

    # 🔹 ถ้ามีคำตอบอยู่แล้วใน intent_responses.json
    if intent in responses:
        return random.choice(responses[intent])

    # 🔹 ถ้า intent ไม่รู้จัก → ใช้ Ollama Llama3.2
    prompt = f"""
    ผู้ใช้: {user_input}
    บริบทก่อนหน้า: {context}

    โปรดตอบกลับเป็นภาษาไทยอย่างสุภาพและเป็นธรรมชาติ
    โดยให้ตอบคล้ายแชทบอทผู้ช่วยทั่วไป
    """

    try:
        # เรียก Ollama CLI เพื่อรันโมเดล llama3.2
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True
        )
        reply = result.stdout.strip()

        if not reply:
            reply = "ขอโทษครับ ระบบไม่สามารถตอบได้ในตอนนี้ 😅"

    except Exception as e:
        reply = f"เกิดข้อผิดพลาดในการเรียกโมเดล: {e}"

    return reply
