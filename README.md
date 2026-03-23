# Qwen3 医疗助手项目

这是一个基于 Qwen3 的医疗领域微调与推理项目，包含数据下载、数据格式转换、全参数微调、LoRA 微调、结构化医学知识生成、基础推理和交互式医疗助手等完整流程。

仓库根目录已经调整为当前目录，下面所有命令默认在仓库根目录执行。

## 主要功能

- 数据集自动下载、划分和格式化
- Qwen3-0.6B 全参数微调
- Qwen3-1.7B LoRA 微调
- 结构化医学知识解释与科普训练
- 基础推理、LoRA 推理、命令行预测、交互式医疗助手
- SwanLab 训练记录，支持本地模式和云端模式

## 仓库结构

```text
.
├── data/
│   ├── data.py
│   ├── transform_medical_dataset.py
│   ├── train.jsonl                  # 运行 data.py 后生成
│   ├── val.jsonl                    # 运行 data.py 后生成
│   ├── train_format.jsonl           # 训练脚本自动生成
│   ├── val_format.jsonl             # 训练脚本自动生成
│   ├── medical_education_structured.jsonl
│   ├── train_structured.jsonl
│   └── val_structured.jsonl
├── inference/
│   ├── inference.py
│   └── inference_lora.py
├── main/
│   ├── medical_assistant.py
│   └── medical_education_assistant.py
├── train/
│   ├── env_utils.py
│   ├── train.py
│   ├── train_lora.py
│   └── train_lora_structured.py
├── download_model.py
├── predict.py
├── replay_test3.py
├── requirements.txt
├── .env.example
├── README.md
└── README_EN.md
```

## 环境要求

- Python 3.10 或更高版本
- 有 CUDA 的显卡更适合训练；没有 GPU 也可以跑推理或小规模验证
- 依赖包见 `requirements.txt`

安装依赖：

```bash
pip install -r requirements.txt
```

## 环境变量

训练脚本会自动读取仓库根目录下的 `.env` 文件。建议先复制示例文件：

```bash
copy .env.example .env
```

然后按需填写：

- `SWANLAB_API_KEY`
- `SWANLAB_MODE`
- `SWANLAB_SAVE_DIR`
- `SWANLAB_LOG_DIR`
- `SWANLAB_PROJECT`

说明：

- 如果设置了 `SWANLAB_API_KEY`，默认走 `cloud`
- 如果没有设置 `SWANLAB_API_KEY`，默认走 `local`
- `SWANLAB_SAVE_DIR` 默认是仓库根目录下的 `.swanlab/`
- `SWANLAB_LOG_DIR` 默认是仓库根目录下的 `swanlog/`
- `SWANLAB_PROJECT` 不填时由训练脚本使用默认项目名

## 数据准备

### 1. 下载并划分原始数据

```bash
python data\data.py
```

这一步会下载 `krisfu/delicate_medical_r1_data`，并生成：

- `data\train.jsonl`
- `data\val.jsonl`

### 2. 生成结构化医学科普数据

```bash
python data\transform_medical_dataset.py
```

这一步会基于 `data\train_format.jsonl` 生成：

- `data\medical_education_structured.jsonl`
- `data\train_structured.jsonl`
- `data\val_structured.jsonl`

## 训练

### 全参数微调

```bash
python train\train.py
```

说明：

- 使用 Qwen3-0.6B
- 读取 `data\train.jsonl` 和 `data\val.jsonl`
- 自动生成 `data\train_format.jsonl` 和 `data\val_format.jsonl`
- 输出目录为 `output\Qwen3-0.6B`

### LoRA 微调

```bash
python train\train_lora.py
```

说明：

- 使用 Qwen3-1.7B
- 读取 `data\train.jsonl` 和 `data\val.jsonl`
- 自动生成 `data\train_format.jsonl` 和 `data\val_format.jsonl`
- 输出目录为 `output\Qwen3-1.7B`

### 结构化医学知识科普训练

```bash
python train\train_lora_structured.py
```

说明：

- 使用结构化数据 `data\train_structured.jsonl` 和 `data\val_structured.jsonl`
- 输出目录为 `output\Qwen3-0.6B-structured`
- 适合训练“结构化解释与科普”风格的模型

## 推理与助手

### 基础推理

```bash
python inference\inference.py
```

说明：

- 会下载并缓存 Qwen3-0.6B 到 `models\`
- 适合验证基础模型的单轮回答效果

### LoRA 推理

```bash
python inference\inference_lora.py
```

说明：

- 会下载并缓存 Qwen3-0.6B 到 `models\`
- 需要你把脚本里 `PeftModel.from_pretrained(...)` 的 LoRA 路径改成自己训练得到的 checkpoint
- 这是一个模板脚本，不是开箱即用的固定路径

### 命令行预测

```bash
python predict.py --input "医生，我最近头痛，可能是什么原因？"
```

说明：

- 会自动寻找 `output\Qwen3-0.6B` 下最新的 `checkpoint-*`
- 可以用 `-c` 显式指定 checkpoint
- 可以用 `-s` 替换 system 提示词
- 可以用 `-m` 调整生成长度

常见参数：

- `--input` 或 `-i`：问题文本
- `--instruction` 或 `-s`：system 提示词
- `--checkpoint` 或 `-c`：模型 checkpoint 路径
- `--max_new_tokens` 或 `-m`：最大生成长度

### 交互式医疗助手

```bash
python main\medical_assistant.py -c output\Qwen3-1.7B\checkpoint-xxxx
```

说明：

- 支持 10 个医疗场景
- 支持单次问答、交互式问答和批量 JSON 问题文件
- 建议通过 `-c` 显式指定你自己的 checkpoint 路径

常见参数：

- `--checkpoint` 或 `-c`：模型 checkpoint
- `--question` 或 `-q`：直接提问
- `--scenario` 或 `-s`：场景类型
- `--max-tokens` 或 `-m`：最大生成 token 数
- `--batch` 或 `-b`：批量问题文件
- `--save-history`：保存对话历史

可用场景：

- `diagnosis`
- `treatment`
- `prevention`
- `education`
- `emergency`
- `nutrition`
- `mental_health`
- `pediatric`
- `geriatric`
- `women_health`

### 结构化医学知识解释助手

```bash
python main\medical_education_assistant.py -c output\Qwen3-0.6B-structured
```

说明：

- 支持结构化医学知识解释与科普
- 默认会在你传入的目录里自动寻找最新 `checkpoint-*`
- 也支持单次问答和批量 JSON 问题文件

可用场景：

- `general`
- `concept`
- `drug_test`
- `disease`
- `health`

## 其他脚本

- `download_model.py`：下载 Qwen3-1.7B 到本地 `models\`
- `replay_test3.py`：读取 `data\val_format.jsonl` 的前 3 条样本做回放测试
- `README_EN.md`：英文说明文档

## 输出目录

运行后可能生成以下本地目录或文件：

- `models\`
- `output\`
- `.swanlab\`
- `swanlog\`
- `data\train.jsonl`
- `data\val.jsonl`
- `data\train_format.jsonl`
- `data\val_format.jsonl`
- `data\medical_education_structured.jsonl`
- `data\train_structured.jsonl`
- `data\val_structured.jsonl`

这些都是运行时产物，公开仓库里通常不需要提交。

## 使用建议

- 公共仓库不要提交 `.env`
- 不要提交真实密钥、token、密码
- 不要提交大模型权重和训练产物，除非你明确希望公开
- 如果原始数据包含敏感信息，先做脱敏再公开

## 模型与数据来源

- 基础模型：Qwen3-0.6B、Qwen3-1.7B
- 数据集：`krisfu/delicate_medical_r1_data`
- 训练监控：SwanLab

## 许可证和引用

如果你打算公开发布，请确认原始项目、模型和数据集的许可证要求，并在需要时补充作者署名和引用说明。

