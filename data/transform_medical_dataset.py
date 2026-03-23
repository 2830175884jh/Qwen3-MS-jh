# -*- coding: utf-8 -*-
import json
import random
import re
from pathlib import Path

# =========================
# 固定路径配置：按你本地情况改这里
# =========================
DATA_DIR = Path(__file__).resolve().parent
SRC_PATH = DATA_DIR / "train_format.jsonl"
OUT_PATH = DATA_DIR / "medical_education_structured.jsonl"
TRAIN_PATH = DATA_DIR / "train_structured.jsonl"
VAL_PATH = DATA_DIR / "val_structured.jsonl"

NEW_INSTRUCTION = "你是一名医学知识解释与科普助手，请针对用户的问题给出结构化、通俗、严谨的解释。不要输出思考过程，不要输出<think>标签。"


def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_think_answer(output: str):
    m = re.search(r"<think>(.*?)</think>\s*(.*)$", output, flags=re.S)
    if m:
        return normalize_text(m.group(1)), normalize_text(m.group(2))
    text = normalize_text(re.sub(r"</?think>", "", output))
    return "", text


def classify_question(text: str) -> str:
    rules = [
        ("药物检测与实验方法说明", ["测定", "含量", "制备", "保存", "培养", "实验", "检测", "测量", "步骤", "试验", "药典"]),
        ("药品与用药说明", ["用药", "药物", "剂量", "服用", "副作用", "不良反应", "禁忌", "药理", "注射", "片剂", "胶囊"]),
        ("检查与检验解读", ["检查", "检验", "化验", "ct", "mri", "b超", "超声", "影像", "指标", "报告", "尿常规", "肝功能", "肾功能"]),
        ("医疗制度与规范说明", ["医院", "购进", "记录", "制度", "规范", "医保", "两票制", "流程", "管理"]),
        ("治疗与处理建议", ["治疗", "处理", "怎么办", "方案", "干预", "缓解", "预防", "护理", "康复"]),
        ("疾病知识与机制解释", ["是什么", "原因", "机制", "病理", "为什么", "发病", "症状", "表现", "关系", "影响"]),
    ]
    tl = text.lower()
    for label, kws in rules:
        if any(kw.lower() in tl for kw in kws):
            return label
    return "医学知识解释"


def split_sentences_cn(text: str):
    text = normalize_text(text)
    parts = re.split(r"(?<=[。！？；])\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def clean_prefix(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(您好|你好|医生|答|患者您好)[，,:：\s]*", "", text)
    return text.strip()


def first_meaningful_sentence(text: str) -> str:
    parts = split_sentences_cn(clean_prefix(text))
    if parts:
        return parts[0]
    return clean_prefix(text)[:120].strip()


def build_plain_explanation(answer: str) -> str:
    sents = split_sentences_cn(clean_prefix(answer))
    if not sents:
        return ""
    tail = sents[1:3]
    if tail:
        return " ".join(tail)
    return sents[0]


def infer_tip(question: str, answer: str, qtype: str) -> str:
    q = question.lower()
    a = answer.lower()
    if any(k in q + a for k in ["实验", "检测", "测定", "培养", "保存", "操作步骤", "药典"]):
        return "实验或检测操作应以药典、指南或实验室标准流程为准。"
    if any(k in q + a for k in ["检查", "检验", "化验", "ct", "mri", "b超", "超声", "报告", "指标"]):
        return "检查结果应结合临床症状、体征及医生判断综合解读。"
    if any(k in q + a for k in ["用药", "服用", "剂量", "副作用", "不良反应", "禁忌", "相互作用"]):
        return "具体用药应遵医嘱执行，涉及剂量调整或不良反应时不要自行处理。"
    if any(k in q + a for k in ["治疗", "怎么办", "处理", "症状", "疾病", "预防", "护理"]):
        return "若症状持续、加重或出现明显不适，应及时到正规医院就诊。"
    if any(k in q + a for k in ["医院", "购进", "记录", "制度", "规范", "医保", "两票制", "管理"]):
        return "涉及制度、流程或合规要求时，应以最新政策文件和单位规范为准。"
    return "以上内容用于医学知识解释与科普，不能替代医生面诊和正式医疗建议。"


def main():
    records = []
    with SRC_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    seen = set()
    processed = []

    for item in records:
        input_text = normalize_text(item["input"])
        if input_text in seen:
            continue
        seen.add(input_text)

        _, answer = split_think_answer(item["output"])
        qtype = classify_question(input_text + "\n" + answer)
        core = first_meaningful_sentence(answer)
        explain = build_plain_explanation(answer)
        if not explain or explain == core:
            explain = "该问题需要结合具体情境理解，重点应关注定义、原理和实际应用。"
        tip = infer_tip(input_text, answer, qtype)

        new_item = {
            "instruction": NEW_INSTRUCTION,
            "input": input_text,
            "output": (
                f"【问题类型】{qtype}\n"
                f"【核心结论】{core}\n"
                f"【通俗解释】{explain}\n"
                f"【关键原理/依据】回答基于原始医学解答中的关键信息整理而成，重点保留定义、机制、步骤或应用要点。\n"
                f"【温馨提示】{tip}"
            )
        }
        processed.append(new_item)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    random.seed(42)
    random.shuffle(processed)
    split_idx = int(len(processed) * 0.9)
    train_data = processed[:split_idx]
    val_data = processed[split_idx:]

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with VAL_PATH.open("w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"原始数据量: {len(records)}")
    print(f"去重后数据量: {len(processed)}")
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"完整输出: {OUT_PATH}")
    print(f"训练集输出: {TRAIN_PATH}")
    print(f"验证集输出: {VAL_PATH}")


if __name__ == "__main__":
    main()

