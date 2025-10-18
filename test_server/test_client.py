"""
Test Client สำหรับทดสอบ AI Core Service
รัน: python test_client.py
"""
import requests
import json
from colorama import Fore, Style, init

# เปิดใช้งาน colorama
init(autoreset=True)

AI_CORE_URL = "http://localhost:8001"

def print_header():
    """แสดง header"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}🤖 AI Core Test Client")
    print(f"{Fore.CYAN}{'='*60}\n")

def check_health():
    """ตรวจสอบสถานะของ AI Core"""
    try:
        response = requests.get(f"{AI_CORE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"{Fore.GREEN}✅ AI Core is healthy")
            print(f"{Fore.YELLOW}   Status: {data.get('status')}")
            print(f"{Fore.YELLOW}   Model: {data.get('model_loaded')}")
            return True
        else:
            print(f"{Fore.RED}❌ AI Core returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}❌ Cannot connect to AI Core at {AI_CORE_URL}")
        print(f"{Fore.YELLOW}   Make sure AI Core is running:")
        print(f"{Fore.YELLOW}   uvicorn ai_core:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}")
        return False

def send_message(message, user_id="test_user", platform="test"):
    """ส่งข้อความไปยัง AI Core"""
    try:
        payload = {
            "message": message,
            "user_id": user_id,
            "platform": platform
        }
        
        print(f"\n{Fore.BLUE}📤 Sending: {Fore.WHITE}{message}")
        
        response = requests.post(
            f"{AI_CORE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{Fore.GREEN}📥 Response: {Fore.WHITE}{data['response']}")
            print(f"{Fore.YELLOW}   Intent: {data.get('intent', 'unknown')}")
            print(f"{Fore.YELLOW}   Confidence: {data.get('confidence', 0.0):.2f}")
            
            # แสดงข้อมูลเพิ่มเติม (ถ้ามี)
            if data.get('data'):
                print(f"{Fore.CYAN}   Data: {json.dumps(data['data'], ensure_ascii=False, indent=2)}")
            
            return data
        else:
            print(f"{Fore.RED}❌ Error: Status {response.status_code}")
            print(f"{Fore.RED}   {response.text}")
            return None
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}")
        return None

def run_test_scenarios():
    """รันชุดการทดสอบ"""
    print(f"\n{Fore.CYAN}🧪 Running Test Scenarios...")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    test_cases = [
        ("สวัสดี", "greeting"),
        ("มีเสื้อยืดสีขาวไหม", "ask_info"),
        ("ขอดูรองเท้าผ้าใบ", "ask_info"),
        ("ราคากางเกงยีนส์เท่าไหร่", "ask_info"),
        ("ต้องการสั่งซื้อสินค้า", "order_product"),
        ("ต้องการคืนสินค้า", "refund_request"),
        ("ช่วยเหลือด้วยค่ะ", "help_request"),
        ("ขอบคุณมากค่ะ", "thank_you"),
    ]
    
    results = []
    for message, expected_intent in test_cases:
        response = send_message(message)
        if response:
            actual_intent = response.get('intent')
            match = "✅" if actual_intent == expected_intent else "❌"
            results.append({
                "message": message,
                "expected": expected_intent,
                "actual": actual_intent,
                "match": match
            })
        print()
    
    # สรุปผล
    print(f"\n{Fore.CYAN}📊 Test Results Summary")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    for result in results:
        color = Fore.GREEN if result['match'] == "✅" else Fore.RED
        print(f"{color}{result['match']} {result['message']}")
        print(f"   Expected: {result['expected']}, Got: {result['actual']}")
    
    correct = sum(1 for r in results if r['match'] == "✅")
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{Fore.YELLOW}Accuracy: {correct}/{total} ({accuracy:.1f}%)")

def interactive_mode():
    """โหมดแชทแบบ Interactive"""
    print(f"\n{Fore.CYAN}💬 Interactive Chat Mode")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}Type 'quit' or 'exit' to stop\n")
    
    user_id = f"test_user_{requests.get('http://localhost:8001').elapsed.microseconds}"
    
    while True:
        try:
            message = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            if message.lower() in ['quit', 'exit', 'q']:
                print(f"{Fore.YELLOW}Goodbye! 👋")
                break
            
            if not message.strip():
                continue
            
            send_message(message, user_id=user_id)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Goodbye! 👋")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")

def main():
    """Main function"""
    print_header()
    
    # ตรวจสอบสถานะ AI Core
    if not check_health():
        return
    
    # เมนู
    while True:
        print(f"\n{Fore.CYAN}เลือกโหมดการทดสอบ:")
        print(f"{Fore.WHITE}1) Run Test Scenarios (ทดสอบอัตโนมัติ)")
        print(f"{Fore.WHITE}2) Interactive Chat (แชทโต้ตอบ)")
        print(f"{Fore.WHITE}3) Single Message Test")
        print(f"{Fore.WHITE}0) Exit")
        
        choice = input(f"\n{Fore.YELLOW}เลือก (0-3): {Style.RESET_ALL}")
        
        if choice == "1":
            run_test_scenarios()
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            message = input(f"{Fore.GREEN}ข้อความ: {Style.RESET_ALL}")
            if message.strip():
                send_message(message)
        elif choice == "0":
            print(f"{Fore.YELLOW}Goodbye! 👋")
            break
        else:
            print(f"{Fore.RED}Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Goodbye! 👋")