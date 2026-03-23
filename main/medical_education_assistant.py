#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化医学知识解释与科普助手
基于 Qwen3-0.6B-mededu-structured LoRA 微调结果
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM
except Exception:
    AutoPeftModelForCausalLM = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 更贴合你这次“结构化医学知识解释与科普”数据的提示词
MEDICAL_EDU_PROMPTS: Dict[str, str] = {
    "general": (
        "你是一名医学知识解释与科普助手。"
        "请针对用户的问题给出结构化、通俗、严谨的解释。"
        "回答必须尽量使用以下结构："
        "【问题类型】【核心结论】【通俗解释】【关键原理/依据】【温馨提示】。"
        "不要输出思考过程，不要输出<think>标签。"
    ),
    "concept": (
        "你是一名医学概念解释助手。"
        "请用患者容易理解的语言，结构化解释医学概念。"
        "回答尽量使用："
        "【问题类型】【核心结论】【通俗解释】【关键原理/依据】【温馨提示】。"
        "不要输出思考过程，不要输出<think>标签。"
    ),
    "drug_test": (
        "你是一名药物与检查说明助手。"
        "请对药物、检查、检测方法相关问题进行结构化解释。"
        "回答尽量使用："
        "【问题类型】【核心结论】【通俗解释】【关键原理/依据】【温馨提示】。"
        "不要输出思考过程，不要输出<think>标签。"
    ),
    "disease": (
        "你是一名疾病知识科普助手。"
        "请针对疾病相关问题做结构化、通俗解释。"
        "回答尽量使用："
        "【问题类型】【核心结论】【通俗解释】【关键原理/依据】【温馨提示】。"
        "不要输出思考过程，不要输出<think>标签。"
    ),
    "health": (
        "你是一名健康科普助手。"
        "请围绕健康管理、预防、生活方式等问题，输出结构化科普内容。"
        "回答尽量使用："
        "【问题类型】【核心结论】【通俗解释】【关键原理/依据】【温馨提示】。"
        "不要输出思考过程，不要输出<think>标签。"
    ),
}

MEDICAL_EDU_SCENARIOS: Dict[str, Tuple[str, str]] = {
    "1": ("general", "综合医学解释"),
    "2": ("concept", "医学概念解释"),
    "3": ("drug_test", "药物/检查说明"),
    "4": ("disease", "疾病知识科普"),
    "5": ("health", "健康管理与预防"),
}

SAMPLE_QUESTIONS: Dict[str, List[str]] = {
    "general": [
        "胆汁排泄和肝胆表里关系有什么联系？",
        "长期发热为什么需要做血常规和影像学检查？",
    ],
    "concept": [
        "什么是高血压？为什么有些人没有明显症状？",
        "糖尿病的发病机制可以通俗解释一下吗？",
    ],
    "drug_test": [
        "盐酸甲氧明注射液的含量测定一般怎么做？",
        "为什么有些药物说明书里要强调饭前服用？",
    ],
    "disease": [
        "胃溃疡是怎么形成的？",
        "慢性肾病早期为什么可能没有明显症状？",
    ],
    "health": [
        "如何预防心血管疾病？",
        "日常饮食怎么做才能更有利于控制血糖？",
    ],
}


def find_latest_checkpoint(output_dir: str) -> str:
    """
    自动寻找最新 checkpoint。
    如果 output_dir 本身就是 checkpoint 目录，则直接返回。
    """
    if os.path.isfile(os.path.join(output_dir, "adapter_config.json")):
        return output_dir

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"模型目录不存在: {output_dir}")

    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                checkpoint_dirs.append((step, full))
            except ValueError:
                continue

    if not checkpoint_dirs:
        raise FileNotFoundError(f"在目录中未找到 checkpoint: {output_dir}")

    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs[-1][1]


def post_process_response(text: str) -> str:
    """
    清理模型输出：
    1. 删除完整 <think>...</think>
    2. 若出现未闭合 <think>，从其开始截断
    3. 截断 Human:/A:/特殊模板残留
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    if "<think>" in text:
        text = text.split("<think>")[0].strip()

    stop_markers = [
        "\nHuman:",
        "Human:",
        "\nA:",
        "\nQ:",
        "<|im_start|>",
        "<|im_end|>",
    ]
    stop_pos = len(text)
    for marker in stop_markers:
        pos = text.find(marker)
        if pos != -1:
            stop_pos = min(stop_pos, pos)

    text = text[:stop_pos].strip()

    # 去掉多余空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


class MedicalEducationAssistant:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.default_output_dir = os.path.join(self.project_root, "output", "Qwen3-0.6B-structured")
        self.checkpoint_path = checkpoint_path or self.default_output_dir
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history: List[dict] = []

    def _select_device_and_dtype(self) -> Tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32

    def load_model(self) -> None:
        print("正在加载结构化医学知识解释模型...")

        resolved_path = find_latest_checkpoint(self.checkpoint_path)
        self.checkpoint_path = resolved_path
        print(f"使用 checkpoint: {self.checkpoint_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loaded = False
        load_errors = []

        # 优先尝试 PEFT 加载
        if AutoPeftModelForCausalLM is not None:
            try:
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                loaded = True
            except Exception as e:
                load_errors.append(f"AutoPeftModelForCausalLM 加载失败: {e}")

        # 回退到普通 AutoModel
        if not loaded:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                loaded = True
            except Exception as e:
                load_errors.append(f"AutoModelForCausalLM 加载失败: {e}")

        if not loaded:
            raise RuntimeError("模型加载失败：\n" + "\n".join(load_errors))

        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成！设备: {self.device}")
        print(f"模型类型: {type(self.model)}")

    def predict(self, messages: List[Dict[str, str]], max_new_tokens: int = 384) -> str:
        model_device = next(self.model.parameters()).device

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = (
            inputs.attention_mask.to(model_device)
            if hasattr(inputs, "attention_mask")
            else None
        )

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只解码新生成部分
        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        response = post_process_response(response)
        return response

    def ask_question(self, question: str, scenario_type: str = "general", max_tokens: int = 384) -> str:
        if scenario_type not in MEDICAL_EDU_PROMPTS:
            scenario_type = "general"

        messages = [
            {"role": "system", "content": MEDICAL_EDU_PROMPTS[scenario_type]},
            {"role": "user", "content": question},
        ]

        self.conversation_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scenario": scenario_type,
                "question": question,
                "response": None,
            }
        )

        response = self.predict(messages, max_new_tokens=max_tokens)
        self.conversation_history[-1]["response"] = response
        return response

    def show_scenarios(self) -> None:
        print("\n🏥 结构化医学知识解释助手 - 可用场景")
        print("=" * 52)
        for key, (_, label) in MEDICAL_EDU_SCENARIOS.items():
            print(f"{key:>2}. {label}")
        print("=" * 52)

    def show_sample_questions(self, scenario_type: str) -> None:
        if scenario_type in SAMPLE_QUESTIONS:
            print("\n📋 示例问题：")
            print("-" * 40)
            for i, question in enumerate(SAMPLE_QUESTIONS[scenario_type], 1):
                print(f"{i}. {question}")
            print("-" * 40)

    def show_help(self) -> None:
        print("\n📖 使用帮助")
        print("=" * 50)
        print("1. 选择一个医学解释场景")
        print("2. 输入你的问题")
        print("3. 系统会尽量按结构化格式输出")
        print("\n💡 说明")
        print("- 本系统主要用于医学知识解释与健康科普")
        print("- 不能替代医生面诊和正式诊断")
        print("- 如遇急症或危险症状，请及时线下就医")
        print("=" * 50)

    def interactive_mode(self) -> None:
        print("\n🤖 结构化医学知识解释助手已启动！")
        print("输入 'help' 查看帮助，输入 'quit' 退出")

        while True:
            try:
                self.show_scenarios()

                scenario_choice = input("\n请选择场景 (1-5): ").strip()
                if scenario_choice == "quit":
                    break
                if scenario_choice == "help":
                    self.show_help()
                    continue
                if scenario_choice not in MEDICAL_EDU_SCENARIOS:
                    print("❌ 无效选择，请重新输入")
                    continue

                scenario_type, scenario_label = MEDICAL_EDU_SCENARIOS[scenario_choice]
                self.show_sample_questions(scenario_type)

                question = input(f"\n请输入您的{scenario_label}问题: ").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue

                print("\n🔄 正在生成结构化解释...")
                start_time = time.time()
                response = self.ask_question(question, scenario_type)
                elapsed = time.time() - start_time

                print(f"\n💡 助手回答 (耗时: {elapsed:.2f}秒)")
                print("=" * 60)
                print(response if response else "未生成有效回答，请尝试换一种问法。")
                print("=" * 60)

                continue_choice = input("\n是否继续咨询？(y/n): ").strip().lower()
                if continue_choice in ["n", "no", "否"]:
                    break

            except KeyboardInterrupt:
                print("\n\n👋 已退出。")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

    def save_conversation(self, filename: Optional[str] = None) -> None:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_education_conversation_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

        print(f"💾 对话历史已保存到: {filename}")

    def batch_questions(self, questions_file: str) -> None:
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)

        print(f"📝 开始批量处理 {len(questions)} 个问题...")

        results = []
        for i, q in enumerate(questions, 1):
            print(f"\n处理第 {i}/{len(questions)} 个问题...")
            scenario = q.get("scenario", "general")
            response = self.ask_question(
                q.get("question", ""),
                scenario,
                q.get("max_tokens", 384),
            )
            results.append(
                {
                    "question": q.get("question", ""),
                    "scenario": scenario,
                    "response": response,
                }
            )

        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ 批量处理完成！结果已保存到: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="结构化医学知识解释与科普助手（Qwen3-0.6B-structured）"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output", "Qwen3-0.6B-structured"),
        help="模型 checkpoint 目录或其父目录。默认会自动选择最新 checkpoint",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="直接询问一个问题",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="general",
        choices=list(MEDICAL_EDU_PROMPTS.keys()),
        help="解释场景类型",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=2048,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=str,
        help="批量处理问题文件（JSON格式）",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="保存对话历史",
    )

    args = parser.parse_args()

    assistant = MedicalEducationAssistant(args.checkpoint)
    assistant.load_model()

    if args.batch:
        assistant.batch_questions(args.batch)
    elif args.question:
        print("🤖 助手回答")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        assistant.interactive_mode()

    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()

