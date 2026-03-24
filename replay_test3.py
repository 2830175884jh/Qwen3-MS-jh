import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 1. 基本配置
# =========================
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# 你想复现哪个 checkpoint，就改这里
CHECKPOINT_PATH = os.path.join(SCRIPT_PATH, "output", "Qwen3-0.6B-structured", "checkpoint-970")

# 这里要用你训练时生成好的格式化验证集
TEST_FILE = os.path.join(SCRIPT_PATH, "data", "val_format.jsonl")

# 生成长度
MAX_NEW_TOKENS = 4096   # 先别太小，避免只出think没出正文

# 是否打印模型结构简要信息
PRINT_MODEL_INFO = True


# =========================
# 2. 设备选择
# =========================
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32


# =========================
# 3. 加载 tokenizer 和 model
# =========================
print(f"正在加载 checkpoint: {CHECKPOINT_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT_PATH,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True
)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=DTYPE,
    local_files_only=True,
    trust_remote_code=True
).to(DEVICE)

model.eval()

if PRINT_MODEL_INFO:
    print("模型类型：", type(model))
    print("pad_token_id =", tokenizer.pad_token_id)
    print("eos_token_id =", tokenizer.eos_token_id)


# =========================
# 4. 预测函数
# =========================
def predict(messages, model, tokenizer, max_new_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只保留新生成部分
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response.strip()


# =========================
# 5. 读取前3条测试数据
# =========================
test_df = pd.read_json(TEST_FILE, lines=True)[:3]

print("\n================ 开始复现前3条 ================\n")

for index, row in test_df.iterrows():
    instruction = row["instruction"]
    input_value = row["input"]

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_value}
    ]

    response = predict(
        messages=messages,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS
    )

    print(f"----- 第 {index + 1} 条 -----")
    print(f"Question: {input_value}")
    print(f"\nLLM: {response}")
    print("\n")

