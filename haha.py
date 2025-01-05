import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import pyttsx3  # 用於文字轉語音 (Text to Speech) 的套件
import speech_recognition as sr  # 用於語音轉文字 (Speech to Text) 的套件

# ==============================
# 一、定義後端商品清單 (menu)
# ==============================
menu = {
    "item_01": {"name": "【義美】小泡芙-巧克力171g", "category": "snack", "price": 88, "promotion": None},
    "item_02": {"name": "【多力多滋】美式辣起士(108g)", "category": "snack", "price": 36, "promotion": None},
    "item_03": {"name": "【蝦味先】原味超派包115g", "category": "snack", "price": 35, "promotion": None},
    "item_04": {"name": "【77】新貴派-花生(16g*18入)", "category": "snack", "price": 85, "promotion": None},
    "item_05": {"name": "【可樂果】原味(188g)", "category": "snack", "price": 52, "promotion": None},
    "item_06": {"name": "【Glico 格力高】Pocky百奇 巧克力餅乾棒(95g)", "category": "snack", "price": 99, "promotion": None},
    "item_07": {"name": "【LAY’S 樂事】原味洋芋片經濟包(166g)", "category": "snack", "price": 63, "promotion": "second_item_80_off"},
    "item_08": {"name": "【奧利奧】雙倍夾心餅乾-檸檬起司蛋糕口味(393g)", "category": "snack", "price": 89, "promotion": None},
    "item_09": {"name": "【卡迪那】德州薯條茄汁口味(90g)", "category": "snack", "price": 36, "promotion": None},
    "item_10": {"name": "【咔辣姆久】歡樂分享包-勁辣唐辛子95g", "category": "snack", "price": 44, "promotion": None},
    "item_11": {"name": "【天天果園】美國北極熊富士蘋果(8入禮盒_220g/顆)", "category": "fruit", "price": 535, "promotion": None},
    "item_12": {"name": "【天天果園】韓國空運無籽麝香綠葡萄(2盒_500-600g/盒)", "category": "fruit", "price": 355, "promotion": None},
    "item_13": {"name": "【天天果園】空運進口新鮮藍莓(3盒_125g/盒)", "category": "fruit", "price": 471, "promotion": None},
    "item_14": {"name": "【天天果園】台灣卡蜜拉紅肉哈密瓜(2顆禮盒/約4.5斤)", "category": "fruit", "price": 912, "promotion": None},
    "item_15": {"name": "【海陸管家】冷凍草莓_家庭號(1包_1kg/包)", "category": "fruit", "price": 640, "promotion": None},
    "item_16": {"name": "【老協珍】經典烏參佛跳牆 1630g(固形物630g)/盒", "category": "dishes", "price": 599, "promotion": "time_limited_90_off"},
    "item_17": {"name": "【呷七碗】櫻花蝦米糕 700g/盒", "category": "dishes", "price": 359, "promotion": None},
    "item_18": {"name": "【台南晶英】紅燒海鮮羹 1370g/盒", "category": "dishes", "price": 469, "promotion": None},
    "item_19": {"name": "【老協珍】冰糖豬腳 820g(固形物620g)/盒", "category": "dishes", "price": 359, "promotion": None},
    "item_20": {"name": "【郭老師】黃耆紅棗枸杞烏骨雞湯 2200g/包", "category": "dishes", "price": 399, "promotion": None},
    "item_21": {"name": "【小川漁屋】野生北海大干貝2包(500g±10%/包/10~13顆)", "category": "meat", "price": 1479, "promotion": None},
    "item_22": {"name": "【約克街肉鋪】天然放牧無毒黑羽土雞切塊2包(1000g±10%/包)", "category": "meat", "price": 1203, "promotion": None},
    "item_23": {"name": "【海陸管家】超大包美國單骨牛小排(1包_500g/包)", "category": "meat", "price": 499, "promotion": None},
    "item_24": {"name": "【海陸管家】嚴選鮮凍大草蝦(6盒_10-12尾/盒)", "category": "meat", "price": 1053, "promotion": None},
    "item_25": {"name": "【海陸管家】嚴選智利鮭魚切片(共12片_3片/約300g/包)", "category": "meat", "price": 699, "promotion": "buy1_get1_free"},
    "item_26": {"name": "【愛之味】健康的油切分解茶(無糖) (600gx6入)x4組", "category": "drink", "price": 388, "promotion": None},
    "item_27": {"name": "【每朝健康】綠茶650mlx24瓶", "category": "drink", "price": 439, "promotion": "second_item_80_off"},
    "item_28": {"name": "【Coca Cola 可口可樂】易開罐(330ml*6罐)", "category": "drink", "price": 79, "promotion": None},
    "item_29": {"name": "【立頓】奶茶(250ml*6入)", "category": "drink", "price": 40, "promotion": None},
    "item_30": {"name": "【Coca Cola 可口可樂】寶特瓶(2000ml)", "category": "drink", "price": 39, "promotion": None}
}

# 範例：在 menu 外自行補充對應品項(若需要更精準的品項，可以自行再做對應或修改)
# 假設「蘋果 (單顆)」對應商品：可以依照 item_11 內含 8 顆粗略計價
price_per_apple = menu["item_11"]["price"] / 8  # 535 / 8 ~ 66.875
# 假設「雞胸肉 (單塊)」每塊 70 元 (示範用)
price_per_chicken_breast = 70

# ==============================
# 二、載入 Llama 2 模型 (示範)
# ==============================
print("[INFO] Loading Llama 2 Model...")

# 1. 載入 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 2. 載入模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model.resize_token_embeddings(len(tokenizer))
model.eval()

# 3. 建立推論管線 (Pipeline)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=torch.device("cpu"),  # 若有 GPU 可改成 "cuda:0"
    max_length=512,
    temperature=0.2,
    do_sample=False
)

# ==============================
# 三、語音輸入 (Speech to Text)
# ==============================
def speech_to_text():
    """
    透過麥克風收音，使用 speech_recognition 辨識語音並回傳文字。
    需先安裝： pip install SpeechRecognition pyaudio
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("[INFO] 說話中，請稍候...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="zh-TW")
        print(f"[INFO] 辨識結果：{text}")
    except Exception as e:
        print("[ERROR] 語音辨識失敗：", e)
        text = ""
    return text

# ==============================
# 四、使用 Llama 2 分析購買需求
# ==============================
import re
import json

def parse_order_with_llama2(user_input: str, pipe):
    """
    使用 Llama 2 來解析「蘋果」與「雞胸肉」的數量，並回傳 JSON 格式的結果。
    
    - `pipe` 為已載入的 pipeline("text-generation", ...) 或其他可生成文本的推論函式。
    - 回傳格式範例：
      {
        "items": [
          {"name": "蘋果", "quantity": 2},
          {"name": "雞胸肉", "quantity": 3}
        ]
      }
    若沒成功擷取，或格式解析失敗，則回傳空的 { "items": [] }。
    """

    # ---- 系統指令 (system prompt) ----
    system_prompt = """你是一個擅長解析語句的智慧助理。
從使用者輸入中找出想購買的品項和數量，只能輸出 JSON 格式。
想買的品項可能包含：蘋果、雞胸肉；也可能其他品項(請直接忽略)。
下面是 JSON 回傳格式範例，請完整依照格式輸出，不要多餘說明：

{
  "items": [
    { "name": "蘋果", "quantity": 2 },
    { "name": "雞胸肉", "quantity": 3 }
  ]
}

如使用者沒有提到可解析的品項，就輸出：
{
  "items": []
}
切記：只能輸出上述 JSON，不能包含任何其他說明或文字。
"""

    # ---- 使用者輸入 (user prompt) ----
    user_prompt = f"使用者的輸入：{user_input}"

    # ---- 組合要給模型的最終 Prompt ----
    full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()

    # ---- 呼叫 Llama 2 進行推論 ----
    response = pipe(
        full_prompt,
        max_new_tokens=256,
        temperature=0.2,    # 溫度降低，避免生成亂碼
        do_sample=False     # 關閉隨機抽樣，盡量取最可能的序列
    )

    # pipeline 回傳的結果通常是一個 list，內含一個 dict
    # text-generation pipeline 的預設鍵值為 "generated_text"
    generated_text = response[0]["generated_text"]

    # ---- 嘗試透過 Regex 擷取 JSON ----
    match = re.search(r"\{[\s\S]+\}", generated_text)
    if match:
        json_str = match.group()
        # ---- 嘗試用 json.loads 解析 ----
        try:
            result = json.loads(json_str)
            # 如果成功，直接回傳
            return result
        except json.JSONDecodeError:
            # 解析失敗則回傳空結果
            return {"items": []}
    else:
        # 如果模型沒輸出任何大括號包起來的內容
        return {"items": []}

# ==============================
# 五、結算購物車
# ==============================
def calculate_cart(items_list):
    """
    根據 parse_order_with_llama2() 回傳的內容，
    去匹配後端商品或用預設單價，計算總金額。
    """
    cart = []
    total = 0
    
    for item in items_list:
        name = item["name"]
        quantity = item["quantity"]
        
        # 依照 name 做對應計價 (示範)
        # 蘋果 (單顆) : price_per_apple
        # 雞胸肉 (單塊) : price_per_chicken_breast
        if "蘋果" in name:
            subtotal = price_per_apple * quantity
            cart.append({"item_name": "蘋果(單顆)", "quantity": quantity, "subtotal": subtotal})
            total += subtotal
        elif "雞胸肉" in name:
            subtotal = price_per_chicken_breast * quantity
            cart.append({"item_name": "雞胸肉(單塊)", "quantity": quantity, "subtotal": subtotal})
            total += subtotal
        else:
            # 其他情況: 目前示範不計價，或自行加上更多對應
            cart.append({"item_name": name, "quantity": quantity, "subtotal": 0})
    
    return cart, total

# ==============================
# 六、語音輸出 (Text to Speech)
# ==============================
def text_to_speech(text):
    """
    使用 pyttsx3 將文字轉成語音。
    """
    engine = pyttsx3.init()
    # 可依需求設定說話速度、音量、音色等
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# ==============================
# 七、主流程示範
# ==============================
if __name__ == "__main__":
    # ----------------------------------------------
    # 1) 從麥克風收音做語音轉文字
    # ----------------------------------------------
    # user_speech_text = speech_to_text()
    #
    # 這邊為了測試方便，直接用使用者給的範例字串：
    user_speech_text = "我要蘋果2顆、雞胸肉3塊"

    # ----------------------------------------------
    # 2) 透過 Llama 2 解析欲購買品項與數量
    # ----------------------------------------------
    parse_result = parse_order_with_llama2(user_speech_text,pipe)
    print("[INFO] Llama 2 解析結果：", parse_result)

    # ----------------------------------------------
    # 3) 建立購物車並計算總金額
    # ----------------------------------------------
    cart, total = calculate_cart(parse_result["items"])
    print("[INFO] 購物車內容：", cart)
    print("[INFO] 總金額：", total)

    # ----------------------------------------------
    # 4) 將結果語音輸出
    # ----------------------------------------------
    if cart:
        cart_str_list = []
        for c in cart:
            cart_str_list.append(f"{c['item_name']} {int(c['quantity'])}個，小計 {int(c['subtotal'])}元")
        cart_readout = "，".join(cart_str_list)
        output_text = f"您的購物車有：{cart_readout}。總金額為 {int(total)} 元。"
    else:
        output_text = "很抱歉，沒有辨識到任何可購買的品項。"
    
    print("[INFO] 語音播報：", output_text)
    text_to_speech(output_text)
