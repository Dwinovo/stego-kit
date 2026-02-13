# StegoKit

StegoKit 是一个面向大语言模型（LLM）的生成式隐写工具包，提供统一的编码/解码调度接口，并内置多种经典隐写策略实现。

项目核心目标：
- 统一不同隐写算法的调用方式（同一套 `embed` / `extract` API）
- 便于研究和复现实验（可控随机源、可切换策略、可扩展注册）
- 兼容 Hugging Face `transformers` 因果语言模型推理流程

## 特性

- 统一调度器：`StegoDispatcher`
- 内置 8 种算法策略：`AC / DISCOP / DISCOP_BASE / METEOR / ASYMMETRIC / DIFFERENTIAL_BASED / BINARY_BASED / STABILITY_BASED`
- 标准结果对象：
  - `StegoEncodeResult`（`generated_token_ids`、`consumed_bits`、`text`、`metadata`）
  - `StegoDecodeResult`（`bits`、`metadata`）
- 可插拔随机源：支持传入 `PRG`（`utils/prg.py`）
- 可扩展算法注册：支持自定义策略并通过注册表调用

## 目录结构

```text
stego-kit/
├── main.py                        # 非对称算法演示脚本
├── core/
│   ├── stego_dispatcher.py        # 统一调度入口
│   ├── stego_registry.py          # 算法注册表（内置 + 自定义）
│   ├── stego_context.py           # 编码/解码上下文
│   ├── stego_algorithm.py         # 策略协议与结果结构
│   └── algorithm_enum.py          # 内置算法枚举
├── algo/
│   ├── ac/ac.py
│   ├── discop/discop.py
│   ├── discop/discop_base.py
│   ├── meteor/meteor.py
│   ├── asymmetric/asymmetric.py
│   └── artifacts/
│       ├── differential_based.py
│       ├── binary_based.py
│       └── stability_based.py
├── utils/prg.py                   # HMAC-DRBG 风格 PRG
└── stegokit/__init__.py           # 对外导出 API
```

## 环境要求

- Python >= 3.10
- 建议使用 CUDA 环境（CPU 也可运行，但速度较慢）

项目代码依赖以下第三方库（当前 `pyproject.toml` 未声明运行时依赖，请手动安装）：
- `torch`
- `transformers`
- `numpy`

## 安装

```bash
# 1) 克隆项目
git clone <your-repo-url>
cd stego-kit

# 2) 安装依赖（按你的 CUDA 版本选择 torch 安装方式）
pip install torch transformers numpy

# 3) 安装本项目（开发模式）
pip install -e .
```

## 快速开始

### 1) 基础用法（统一 API）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegokit import StegoAlgorithm, StegoDispatcher, PRG

model_name = "gpt2"  # 示例模型，实际可替换为你自己的 CausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

dispatcher = StegoDispatcher(verbose=True)
prg = PRG.from_int_seed(2026)

enc = dispatcher.embed(
    algorithm=StegoAlgorithm.AC,
    model=model,
    tokenizer=tokenizer,
    secret_bits="010101001011",
    prompt="Write a short paragraph about privacy.",
    max_new_tokens=64,
    temperature=1.0,
    top_k=50,
    precision=16,
    prg=prg,  # AC 可不传；部分算法必须传（见下表）
)

# 解码时必须保持同样的模型与采样参数
dec = dispatcher.extract(
    algorithm=StegoAlgorithm.AC,
    model=model,
    tokenizer=tokenizer,
    generated_token_ids=enc.generated_token_ids,
    prompt="Write a short paragraph about privacy.",
    temperature=1.0,
    top_k=50,
    precision=16,
    max_bits=enc.consumed_bits,
    prg=PRG.from_int_seed(2026),
)

print("generated:", enc.text)
print("embedded bits:", enc.consumed_bits)
print("decoded bits:", dec.bits)
```

### 2) 运行项目内置 demo（非对称算法）

`main.py` 默认读取固定模型路径：

```python
MODEL_PATH = "/root/autodl-fs/Meta-Llama-3-8B-Instruct/"
```

请先按你的环境修改该路径，再运行：

```bash
python main.py
```

## 内置算法一览

| 算法 | 枚举值 | 是否要求 `prg` | 说明 |
|---|---|---|---|
| Arithmetic Coding | `ac` | 否 | 基于区间编码的隐写策略 |
| Discop | `discop` | 是（`generate_random`） | 基于 Huffman 树与随机路径 |
| Discop Base | `discop_base` | 是（`generate_random`） | Discop 的基线版本 |
| Meteor | `meteor` | 是（`generate_bits`） | 掩码比特参与编码/解码 |
| Asymmetric | `asymmetric` | 否 | 支持 `regular` 与 `robust` 两种解码 |
| Differential Based | `differential_based` | 是（`generate_random`） | artifacts 策略之一 |
| Binary Based | `binary_based` | 是（`generate_random`） | artifacts 策略之一 |
| Stability Based | `stability_based` | 是（`generate_random`） | artifacts 策略之一 |

## 参数说明

### 通用参数（`embed` / `extract`）

- `algorithm`: `StegoAlgorithm` 枚举或自定义算法名字符串
- `model`, `tokenizer`: Hugging Face 因果 LM 与对应 tokenizer
- `prompt`: 上下文提示词
- `temperature`, `top_k`, `top_p`: 采样控制
- `precision`: 隐写相关精度参数（>0）
- `prg`: 可选伪随机源对象（某些算法必须）
- `stop_on_eos`: 是否在生成到 `eos_token` 时停止（`None` 表示使用算法默认策略）
- `extra`: 算法特定参数字典

### Asymmetric 的 `extra` 参数

- `seed`（默认 `"12345"`）
- `secure_parameter`（默认 `32`）
- `func_type`（默认 `0`，支持 `0/1/2`）
- `use_chat_template`（默认 `False`）
- 解码额外参数：
  - `decode_mode`: `"regular"` 或 `"robust"`
  - `robust_search_window`（默认 `1000`，仅 robust 模式）

## 结果对象

编码返回 `StegoEncodeResult`：
- `generated_token_ids`: 生成的 token id 序列
- `consumed_bits`: 实际嵌入的 bit 数
- `text`: 解码后的文本
- `metadata`: 算法附加信息（如步骤数、内部状态等）

解码返回 `StegoDecodeResult`：
- `bits`: 提取出的 bit 串
- `metadata`: 附加信息（如解码模式、步数等）

## 自定义算法扩展

实现 `StegoStrategy` 协议（`encode` / `decode`），注册后即可被 `dispatcher` 调用。

```python
from stegokit import (
    StegoAlgorithmRegistry,
    StegoDispatcher,
    StegoEncodeResult,
    StegoDecodeResult,
)

class MyStrategy:
    def encode(self, context):
        return StegoEncodeResult(generated_token_ids=[], consumed_bits=0, text="", metadata={"name": "my_algo"})

    def decode(self, context):
        return StegoDecodeResult(bits="", metadata={"name": "my_algo"})

registry = StegoAlgorithmRegistry.default()
registry.register("my_algo", MyStrategy())
dispatcher = StegoDispatcher(registry=registry)

# 之后可直接使用 algorithm="my_algo"
```

## 使用注意事项

- 编码与解码必须保持同一组关键条件：
  - 相同模型与 tokenizer
  - 相同 prompt
  - 相同采样参数（`temperature` / `top_k` / `top_p` / `precision`）
  - 对要求 PRG 的算法，必须使用一致的 PRG 初始化参数
- `secret_bits` 必须是仅包含 `0/1` 的字符串
- 某些算法对分布和精度敏感，`top_k/top_p/temperature` 变化会直接影响可解码性
- 当前项目偏研究与实验用途，建议在你的任务数据上进行鲁棒性评估

## 开发建议

- 若要增加可复现实验，建议固定：随机种子、模型版本、采样参数、prompt 模板
- 可在 `metadata` 中额外记录实验配置，便于后续分析
- 如需发布到 PyPI，建议在 `pyproject.toml` 中补充运行时依赖

## License

MIT License，见 `LICENSE`。
