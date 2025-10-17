import pickle
import pythainlp

# โหลดโมเดล intent
with open("line_Platform/thai_intent_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_intent(text: str):
    """รับข้อความภาษาไทย -> คืนค่า intent"""
    try:
        intent = model.predict([text])[0]
        return intent
    except Exception as e:
        print("Predict error:", e)
        return "unknown"
