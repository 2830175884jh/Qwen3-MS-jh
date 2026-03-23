# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model

# =========================
# 1. SwanLab 环境配置
# =========================

# 当前脚本目录：../Qwen3-MS/train
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：.../Qwen3-MS
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "Qwen3-0.6B-structured")

TRAIN_FILE = os.path.join(DATA_DIR, "train_structured.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val_structured.jsonl")


from env_utils import configure_swanlab

configure_swanlab(default_project="qwen3-medical-education-structured", repo_root=PROJECT_ROOT)

import swanlab

# =========================
# 2. 基本配置
# =========================
MODEL_NAME = "Qwen/Qwen3-0.6B"



PROMPT = "你是一名医学知识讲解与科普助手，请针对用户的问题给出结构化、通俗、严谨的解释。不要输出思考过程，不要输出<think>标签。"

MAX_LENGTH = 1024
MAX_NEW_TOKENS = 384

# 训练参数
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 1e-4
EVAL_STEPS = 100
LOGGING_STEPS = 10
SAVE_STEPS = 200

swanlab.config.update(
    {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "data_max_length": MAX_LENGTH,
        "train_file": TRAIN_FILE,
        "val_file": VAL_FILE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_TRAIN_EPOCHS,
        "output_dir": OUTPUT_DIR,
    }
)

# =========================
# 3. 加载模型
# =========================
model_dir = snapshot_download(MODEL_NAME, cache_dir=MODEL_CACHE_DIR, revision="master")

# =========================
# 4. 加载 tokenizer / model
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=False,
    trust_remote_code=True,
)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()

# =========================
# 5. LoRA 配置
# =========================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

# 可选：打印可训练参数占比
model.print_trainable_parameters()

# =========================
# 6. 数据预处理
# =========================
def process_func(example):
    """
    将结构化数据集转换为模型训练所需格式。
    数据字段格式：
    {
        "instruction": "...",
        "input": "...",
        "output": "[问题类别]...[核心结论]..."
    }
    """
    # 优先使用样本里的 instruction；没有再回退到全局 PROMPT
    instruction_text = example.get("instruction", PROMPT)
    input_text = example["input"]
    output_text = example["output"]

    # 训练时也统一使用 chat template，避免和推理模板不一致
    messages = [
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": input_text},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt = tokenizer(prompt_text, add_special_tokens=False)
    response = tokenizer(output_text, add_special_tokens=False)

    input_ids = prompt["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = prompt["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(prompt["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# =========================
# 7. 读取数据
# =========================
train_df = pd.read_json(TRAIN_FILE, lines=True)
val_df = pd.read_json(VAL_FILE, lines=True)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
eval_dataset = val_ds.map(process_func, remove_columns=val_ds.column_names)

# =========================
# 8. 训练参数
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    gradient_checkpointing=True,
    save_on_each_node=True,
    report_to="swanlab",
    run_name="qwen3-0.6B-structured",
)

# =========================
# 9. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    ),
)

# =========================
# 10. 开始训练
# =========================
trainer.train()

# =========================
# 11. 推理函数（训练后主要查看）
# =========================
def predict(messages, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS):
    device = next(model.parameters()).device

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask if hasattr(model_inputs, "attention_mask") else None,
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

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response

# =========================
# 12. 鐢ㄩ獙璇侀泦鍓?鏉″仛涓昏娴嬭瘯
# =========================
test_df = pd.read_json(VAL_FILE, lines=True)[:3]
test_text_list = []

for index, row in test_df.iterrows():
    instruction = row["instruction"]
    input_value = row["input"]

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_value},
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
Question: {input_value}

LLM:
{response}
"""

    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

try:
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()
except Exception as e:
    print(f"WARN: SwanLab logging skipped: {e}")



