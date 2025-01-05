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
    # 商品清單省略...
}

# 範例：在 menu 外自行補充對應品項(若需要更精準的品項，可以自行再做對應或修改)
price_per_apple = menu["item_11"]["price"] / 8  # 535 / 8 ~ 66.875
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
    # 函數實作省略...
    return text

# ==============================
# 四、使用 Llama 2 分析購買需求
# ==============================
import re
import json

def parse_order_with_llama2(user_input: str, pipe):
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

    user_prompt = f"使用者的輸入：{user_input}"
    full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()

    response = pipe(
        full_prompt,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False
    )

    generated_text = response[0]["generated_text"]

    match = re.search(r"\{[\s\S]+\}", generated_text)
    if match:
        json_str = match.group()
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            return {"items": []}
    else:
        return {"items": []}

# ==============================
# 五、結算購物車
# ==============================
def calculate_cart(items_list):
    # 函數實作省略...
    return cart, total

# ==============================
# 六、語音輸出 (Text to Speech)
# ==============================
def text_to_speech(text):
    # 函數實作省略...

# ==============================
# 七、主流程示範
# ==============================
if __name__ == "__main__":
    user_speech_text = "我要蘋果2顆、雞胸肉3塊"
    parse_result = parse_order_with_llama2(user_speech_text, pipe)
    print("[INFO] Llama 2 解析結果：", parse_result)
    cart, total = calculate_cart(parse_result["items"])
    print("[INFO] 購物車內容：", cart)
    print("[INFO] 總金額：", total)

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
