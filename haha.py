import re
import json
import torch
import pyttsx3  # 文字轉語音
import speech_recognition as sr  # 語音輸入
from transformers import AutoTokenizer, AutoModelForCausalLM

###################################
# 1. 模型初始化
###################################
print("[INFO] 正在載入 Llama3.x 模型...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model.eval()
print("[INFO] 模型載入完成！")

###################################
# 2. 文字轉語音初始化
###################################
print("[INFO] 初始化文字轉語音引擎...")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # 調整語速
tts_engine.setProperty("volume", 0.9)  # 調整音量
print("[INFO] TTS 引擎初始化完成！")

###################################
# 3. 商品菜單與促銷資料
###################################
menu = {
    # 商品清單略
    "item_01": {"name": "蘋果",   "category": "fruit",     "price": 30,  "promotion": "buy1_get1_free"},
    "item_02": {"name": "香蕉",   "category": "fruit",     "price": 25,  "promotion": None},
    "item_03": {"name": "橙子",   "category": "fruit",     "price": 35,  "promotion": None},
    "item_04": {"name": "草莓",   "category": "fruit",     "price": 40,  "promotion": None},
    "item_05": {"name": "西瓜",   "category": "fruit",     "price": 50,  "promotion": None},

    # 蔬菜 (vegetable)
    "item_06": {"name": "胡蘿蔔", "category": "vegetable", "price": 20,  "promotion": "buy1_get1_free"},
    "item_07": {"name": "西蘭花", "category": "vegetable", "price": 30,  "promotion": None},
    "item_08": {"name": "番茄",   "category": "vegetable", "price": 25,  "promotion": None},
    "item_09": {"name": "萵苣",   "category": "vegetable", "price": 15,  "promotion": None},
    "item_10": {"name": "馬鈴薯", "category": "vegetable", "price": 25,  "promotion": None},

    # 肉類 (meat)
    "item_11": {"name": "雞胸肉", "category": "meat",      "price": 70,  "promotion": "second_item_87_off"},
    "item_12": {"name": "豬五花", "category": "meat",      "price": 60,  "promotion": "time_limited_65_off"},
    "item_13": {"name": "牛肋眼", "category": "meat",      "price": 150, "promotion": None},
    "item_14": {"name": "鮭魚排", "category": "meat",      "price": 120, "promotion": None},
    "item_15": {"name": "羊肩排", "category": "meat",      "price": 180, "promotion": "90_percent_off"},

    # 飲品 (drink)
    "item_16": {"name": "礦泉水", "category": "drink",     "price": 15,  "promotion": None},
    "item_17": {"name": "可樂",   "category": "drink",     "price": 20,  "promotion": None},
    "item_18": {"name": "柳橙汁", "category": "drink",     "price": 30,  "promotion": None},
    "item_19": {"name": "牛奶",   "category": "drink",     "price": 25,  "promotion": "buy1_get1_free"},
    "item_20": {"name": "綠茶",   "category": "drink",     "price": 25,  "promotion": "second_item_87_off"},

    # 零食 (snack)
    "item_21": {"name": "洋芋片", "category": "snack",     "price": 40,  "promotion": "time_limited_65_off"},
    "item_22": {"name": "巧克力", "category": "snack",     "price": 45,  "promotion": "90_percent_off"},
    "item_23": {"name": "餅乾",   "category": "snack",     "price": 35,  "promotion": None},
    "item_24": {"name": "爆米花", "category": "snack",     "price": 25,  "promotion": None},
    "item_25": {"name": "軟糖",   "category": "snack",     "price": 30,  "promotion": None},

    # 生活用品 (household)
    "item_26": {"name": "衛生紙", "category": "household", "price": 40,  "promotion": None},
    "item_27": {"name": "洗衣精", "category": "household", "price": 90,  "promotion": None},
    "item_28": {"name": "洗碗精", "category": "household", "price": 45,  "promotion": None},
    "item_29": {"name": "燈泡",   "category": "household", "price": 55,  "promotion": None},
    "item_30": {"name": "洗髮精", "category": "household", "price": 60,  "promotion": "buy1_get1_free"}
}

###################################
# 4. 促銷規則
###################################
def promo_buy1_get1_free(qty, unit_price):
    pay_qty = (qty + 1) // 2
    return pay_qty * unit_price

def promo_second_item_87_off(qty, unit_price):
    total = 0
    for i in range(qty):
        total += unit_price * (0.87 if i % 2 == 1 else 1)
    return round(total, 2)

promotion_algorithms = {
    "buy1_get1_free": promo_buy1_get1_free,
    "second_item_87_off": promo_second_item_87_off,
}

def calculate_price_with_promotion(item_id, quantity):
    info = menu[item_id]
    price = info["price"]
    promo_type = info["promotion"]
    if promo_type in promotion_algorithms:
        return promotion_algorithms[promo_type](quantity, price)
    else:
        return price * quantity

###################################
# 5. Llama 模型解析用戶輸入
###################################
def llama_generate_text(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

def parse_input_with_llama(user_input):
    prompt = (
        f"你是一個購物助理，請解析以下使用者輸入，並轉換為購買清單。\n"
        "請確保輸出僅為 JSON 格式，例如：\n"
        '[{"product": "商品名稱", "quantity": 商品數量}, {"product": "商品名稱", "quantity": 商品數量}]\n'
        "不要添加任何額外解釋。\n"
        f"輸入: \"{user_input}\"\n"
    )
    gen_text = llama_generate_text(prompt)
    print(f"[DEBUG] 模型輸出: {gen_text}")

    # 匹配所有 JSON 字串
    matches = re.findall(r"\[.*?\]", gen_text, re.DOTALL)

    for match in matches:
        try:
            print(f"[DEBUG] 嘗試解析的 JSON 字串: {match}")
            json_str = match.replace("'", '"')
            parsed_items = json.loads(json_str)
            for item in parsed_items:
                item["quantity"] = int(item["quantity"])  # 確保數量是整數
            print(f"[DEBUG] 解析成功的 JSON: {parsed_items}")
            return parsed_items  # 返回成功解析的 JSON 資料
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ERROR] JSON 解碼失敗，嘗試下一個: {e}")
    print("[ERROR] 無法解析任何有效的 JSON 資料")
    return []

###################################
# 6. 結帳邏輯
###################################
def convert_items_to_cart(parsed_items):
    cart = {}
    for item in parsed_items:
        product_name = item.get("product")
        quantity = item.get("quantity", 1)
        for item_id, info in menu.items():
            if info["name"] == product_name:
                cart[item_id] = cart.get(item_id, 0) + quantity
    return cart

def checkout_cart(cart):
    lines = []
    total = 0
    for item_id, qty in cart.items():
        if item_id not in menu:
            continue
        info = menu[item_id]
        subtotal = calculate_price_with_promotion(item_id, qty)
        promo = info.get("promotion", "")
        promo_str = f"({promo})" if promo else ""
        subtotal = round(subtotal)
        lines.append(f"{info['name']} x {qty} {promo_str} => 小計: {subtotal}")
        total += subtotal
    total = round(total)
    lines.append(f"總金額：{total}")
    return "\n".join(lines), total

###################################
# 7. 語音輸入功能
###################################
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        tts_engine.say("請開始說話")
        tts_engine.runAndWait()
        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"[語音輸入] 您說的是: {text}")
            return text
        except sr.UnknownValueError:
            print("[ERROR] 無法識別語音，請再試一次。")
            return ""
        except sr.RequestError as e:
            print(f"[ERROR] 語音服務出錯: {e}")
            return ""
        except sr.WaitTimeoutError:
            print("[ERROR] 語音輸入超時，請再試一次。")
            return ""

###################################
# 8. 主程式入口
###################################
def main():
    print("=== 歡迎使用【Llama3.x + 促銷聊天機器人】 ===")
    print("輸入範例: 我要蘋果2顆、雞胸肉3塊。")
    print("輸入 'exit' 離開。")

    while True:
        mode = input("\n請選擇輸入模式 (1: 鍵盤輸入, 2: 語音輸入): ").strip()
        if mode.lower() in ["exit", "quit"]:
            print("感謝使用，再見！")
            break

        if mode == "1":
            user_input = input("\n您: ").strip()
        elif mode == "2":
            user_input = get_audio_input().strip()
        else:
            print("[ERROR] 無效選項，請輸入 1 或 2。")
            continue

        if not user_input:
            print("[ERROR] 無有效輸入，請再試一次。")
            continue

        parsed_items = parse_input_with_llama(user_input)
        if not parsed_items:
            reply = "抱歉，我無法理解您的輸入，請再試一次。"
            print(f"機器人: {reply}")
            tts_engine.say(reply)
            tts_engine.runAndWait()
            continue

        cart = convert_items_to_cart(parsed_items)
        detail_str, total_amount = checkout_cart(cart)

        reply = f"購物清單如下：\n{detail_str}\n感謝您的購買！"
        print(f"機器人: {reply}")
        tts_engine.say(reply)
        tts_engine.runAndWait()

if __name__ == "__main__":
    main()
