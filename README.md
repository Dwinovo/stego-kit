# StegoKit

StegoKit 是一个面向大语言模型（LLM）的生成式隐写工具包，提供统一的编码/解码调度接口，并内置多种经典隐写策略实现。

项目核心目标：
- 统一不同隐写算法的调用方式（同一套上下文结构 + 分发接口）
- 便于研究和复现实验（可控随机源、可切换策略、可扩展注册）
- 兼容 Hugging Face `transformers` 因果语言模型推理流程；当前要求 tokenizer 支持并配置 `chat_template`

## 特性

- 统一调度器：`StegoDispatcher`
- 内置 10 种算法策略：`AC / ADG / DISCOP / DISCOP_BASE / HUFFMAN / METEOR / ARS / FDPSS_DIFFERENTIAL_BASED / FDPSS_BINARY_BASED / FDPSS_STABILITY_BASED`
- 标准结果对象：
  - `StegoEncodeResult`（`generated_token_ids`、`consumed_bits`、`text`、`encode_time_seconds`、`embedding_capacity`、`metadata`）
  - `StegoDecodeResult`（`bits`、`decode_time_seconds`、`metadata`）
- 可插拔安全材料：支持通过 `RandomnessMaterial` / `BitMaskMaterial` 注入 `PRG`（`stegokit/utils/prg.py`）
- 可扩展算法注册：支持自定义策略并通过注册表调用

## 目录结构

```text
stego-kit/
├── README.md
├── pyproject.toml
└── stegokit/
    ├── __init__.py                # 对外导出 API
    ├── core/
    │   ├── algorithm_config.py    # 内置算法 typed config
    │   ├── algorithm_enum.py      # 内置算法枚举
    │   ├── algorithm_spec.py      # 算法描述与注册元数据
    │   ├── generation_config.py   # 生成参数配置
    │   ├── runtime_context.py     # model/tokenizer/messages/runtime
    │   ├── security_material.py   # typed 安全材料
    │   ├── stego_algorithm.py     # 策略协议与结果结构
    │   ├── stego_context.py       # 编码/解码上下文
    │   ├── stego_dispatcher.py    # 统一调度入口
    │   ├── stego_paradigm.py      # symmetric / asymmetric 分类
    │   └── stego_registry.py      # 算法注册表（内置 + 自定义）
    ├── algo/
    │   ├── ac/ac.py
    │   ├── adg/adg.py
    │   ├── discop/discop.py
    │   ├── discop/discop_base.py
    │   ├── huffman/huffman.py
    │   ├── meteor/meteor.py
    │   ├── ars/ars.py
    │   └── fdpss/
    │       ├── differential_based.py
    │       ├── binary_based.py
    │       └── stability_based.py
    └── utils/prg.py               # HMAC-DRBG 风格 PRG
```

## 环境要求

- Python >= 3.10
- 建议使用 CUDA 环境（CPU 也可运行，但速度较慢）

项目代码依赖以下第三方库：
- `torch`
- `transformers`
- `numpy`

## 安装

```bash
# 1) 克隆项目
git clone <your-repo-url>
cd stego-kit

# 2) 安装本项目（开发模式）
pip install -e .
```

如果你需要自行控制 `torch` 的 CUDA 轮子来源，可以先按你的环境安装 `torch`，再执行 `pip install -e .`。

## 快速开始

### 1) 基础用法（构建 `runtime/config/material` + 分发器）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegokit import (
    ACConfig,
    GenerationConfig,
    NoMaterial,
    RuntimeContext,
    StegoAlgorithm,
    StegoDecodeContext,
    StegoDispatcher,
    StegoEncodeContext,
)

model_name = "<chat-model-with-chat-template>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
if not getattr(tokenizer, "chat_template", None):
    raise ValueError("StegoKit currently expects tokenizer.chat_template to be configured.")

dispatcher = StegoDispatcher(verbose=True)
messages = [{"role": "user", "content": "Write a short paragraph about privacy."}]
runtime = RuntimeContext(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    generation=GenerationConfig(
        max_new_tokens=64,
        temperature=1.0,
        top_k=50,
        precision=16,
    ),
)

enc_ctx = StegoEncodeContext(
    algorithm=StegoAlgorithm.AC,
    runtime=runtime,
    secret_bits="010101001011",
    config=ACConfig(),
    material=NoMaterial(),
)
enc = dispatcher.dispatch_encode(enc_ctx)

# 解码时必须保持同样的模型与采样参数
dec_ctx = StegoDecodeContext(
    algorithm=StegoAlgorithm.AC,
    runtime=RuntimeContext(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        generation=GenerationConfig(
            temperature=1.0,
            top_k=50,
            precision=16,
        ),
    ),
    generated_token_ids=enc.generated_token_ids,
    max_bits=enc.consumed_bits,
    config=ACConfig(),
    material=NoMaterial(),
)
dec = dispatcher.dispatch_decode(dec_ctx)

print("generated:", enc.text)
print("embedded bits:", enc.consumed_bits)
print("encode time(s):", enc.encode_time_seconds)
print("capacity(bit/token):", enc.embedding_capacity)
print("decoded bits:", dec.bits)
print("decode time(s):", dec.decode_time_seconds)
```

### 2) 便捷封装（`embed` / `extract`）

如果你不想手动构建上下文，也可以直接调用便捷封装：

```python
enc = dispatcher.embed(
    algorithm=StegoAlgorithm.AC,
    model=model,
    tokenizer=tokenizer,
    secret_bits="010101001011",
    messages=messages,
    max_new_tokens=64,
    temperature=1.0,
    top_k=50,
    precision=16,
    config=ACConfig(),
    material=NoMaterial(),
)

dec = dispatcher.extract(
    algorithm=StegoAlgorithm.AC,
    model=model,
    tokenizer=tokenizer,
    generated_token_ids=enc.generated_token_ids,
    messages=messages,
    temperature=1.0,
    top_k=50,
    precision=16,
    max_bits=enc.consumed_bits,
    config=ACConfig(),
    material=NoMaterial(),
)
```

## 内置算法一览

| 算法 | 枚举值 | Material 类型 | 说明 |
|---|---|---|---|
| Arithmetic Coding | `ac` | `NoMaterial` | 基于区间编码的隐写策略 |
| ADG | `adg` | `NoMaterial` | 基于接近 0.5/0.5 的递归动态分组进行 bit 嵌入 |
| Discop | `discop` | `RandomnessMaterial` | 基于 Huffman 树与随机路径 |
| Discop Base | `discop_base` | `RandomnessMaterial` | Discop 的基线版本 |
| Huffman | `huffman` | `NoMaterial` | 取 top `2^bit_num` 候选并按 Huffman 前缀码嵌入 bit 串 |
| Meteor | `meteor` | `BitMaskMaterial` | 掩码比特参与编码/解码 |
| ARS | `ars` | `AsymmetricEncodeMaterial` / `AsymmetricDecodeMaterial` | 支持 `regular` 与 `robust` 两种解码 |
| FDPSS Differential Based | `fdpss_differential_based` | `RandomnessMaterial` | FDPSS 系列策略之一 |
| FDPSS Binary Based | `fdpss_binary_based` | `RandomnessMaterial` | FDPSS 系列策略之一 |
| FDPSS Stability Based | `fdpss_stability_based` | `RandomnessMaterial` | FDPSS 系列策略之一 |

说明：FDPSS 系列当前仅接受 `fdpss_differential_based`、`fdpss_binary_based`、`fdpss_stability_based` 这 3 个名称；旧名称 `differential_based` / `binary_based` / `stability_based` 已移除。

## 参数说明

### 通用字段（`StegoEncodeContext` / `StegoDecodeContext`）

- `algorithm`: `StegoAlgorithm` 枚举或自定义算法名字符串
- `runtime`: `RuntimeContext`
- `config`: 算法特定配置对象
- `material`: 安全材料对象
- 编码侧额外字段：
  - `secret_bits`: 待嵌入 bit 串
- 解码侧额外字段：
  - `generated_token_ids`: 待提取的生成 token 序列
  - `max_bits`: 最多提取多少个 bit（可选）

### `RuntimeContext`

- `model`, `tokenizer`: Hugging Face 因果 LM 与对应 tokenizer
- `tokenizer` 需要支持 `apply_chat_template`，且应已配置 `chat_template`
- `messages`: 对话消息列表（每项至少包含 `role`，并包含 `content` 或 `tool_calls`）
- `generation`: `GenerationConfig`

### `GenerationConfig`

- `temperature`, `top_k`, `top_p`: 采样控制
- `precision`: 隐写相关精度参数（>0）
- `stop_on_eos`: 是否在生成到 `eos_token` 时停止
- `max_new_tokens`: 最多生成多少个 token（仅编码侧实际使用）

说明：`embed` / `extract` 仍可用，它们会自动构造 `RuntimeContext`，也支持直接传入 `config` 与 `material`。

### ARS 的 `config` 类型

- 编码：`ARSEncodeConfig`
  - `seed`（默认 `"12345"`）
  - `secure_parameter`（默认 `32`）
  - `func_type`（默认 `0`，支持 `0/1/2`）
- 解码：`ARSDecodeConfig`
  - 继承编码配置字段
  - `decode_mode`: `"regular"` 或 `"robust"`
  - `robust_search_window`（默认 `1000`，仅 robust 模式）

### ADG 的 `config` 类型

- `ADGConfig`
  - `epsilon`: 判断分组是否足够接近 `0.5/0.5` 的阈值，默认 `0.01`
  - `max_bit`: 单个 token 最多可消费的 bit 数，默认 `15`

### Huffman 的 `config` 类型

- `HuffmanConfig`
  - `bit_num`: 候选集合大小控制参数，实际候选数为 `2 ** bit_num`
  - `candidate_count`: 直接指定候选数；若提供则优先于 `bit_num`
  - 若两者都不传，当前默认使用 `8` 个候选 token

### 常见 `material` 类型

- `NoMaterial`: 不需要额外安全材料
- `RandomnessMaterial(prg=PRG.from_int_seed(...))`: 适用于 `Discop` / `FDPSS` 等依赖 `generate_random` 的算法
- `BitMaskMaterial(prg=PRG.from_int_seed(...))`: 适用于 `Meteor`
- `AsymmetricEncodeMaterial` / `AsymmetricDecodeMaterial`: 适用于 `ARS` 和未来的非对称隐写算法

## 结果对象

编码返回 `StegoEncodeResult`：
- `generated_token_ids`: 生成的 token id 序列
- `consumed_bits`: 实际嵌入的 bit 数
- `text`: 生成出的文本（由 `generated_token_ids` 解码得到）
- `encode_time_seconds`: 编码耗时（秒）
- `embedding_capacity`: 平均每个生成 token 嵌入 bit 数（`consumed_bits / len(generated_token_ids)`）
- `metadata`: 算法附加信息（如步骤数、内部状态等）

解码返回 `StegoDecodeResult`：
- `bits`: 提取出的 bit 串
- `decode_time_seconds`: 解码耗时（秒）
- `metadata`: 附加信息（如解码模式、步数等）

## 自定义算法扩展

实现 `StegoStrategy` 协议（`encode` / `decode`），并注册 `AlgorithmSpec` 后即可被 `dispatcher` 调用。

```python
from stegokit import (
    AlgorithmSpec,
    NoConfig,
    NoMaterial,
    StegoParadigm,
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
registry.register_spec(
    AlgorithmSpec(
        name="my_algo",
        paradigm=StegoParadigm.SYMMETRIC,
        strategy=MyStrategy(),
        encode_config_type=NoConfig,
        decode_config_type=NoConfig,
        encode_material_type=NoMaterial,
        decode_material_type=NoMaterial,
    )
)
dispatcher = StegoDispatcher(registry=registry)

# 之后可直接使用 algorithm="my_algo"
```

## 使用注意事项

- 编码与解码必须保持同一组关键条件：
  - 相同模型与 tokenizer
  - 相同 messages（内容与顺序）
  - 相同采样参数（`temperature` / `top_k` / `top_p` / `precision`）
  - 对要求随机材料的算法，必须使用一致的 `material` 初始化参数
- `secret_bits` 必须是仅包含 `0/1` 的字符串
- 某些算法对分布和精度敏感，`top_k/top_p/temperature` 变化会直接影响可解码性
- 当前项目偏研究与实验用途，建议在你的任务数据上进行鲁棒性评估

## 开发建议

- 若要增加可复现实验，建议固定：随机种子、模型版本、采样参数、messages 模板
- 可在 `metadata` 中额外记录实验配置，便于后续分析

## 论文引用

如果你在研究或工程工作中使用了本项目，也欢迎同时引用对应算法的原始论文。当前内置算法与论文的对应关系如下：

| 算法 | 对应论文 |
|---|---|
| `ac` | Ziegler et al., *Neural Linguistic Steganography* (2019) |
| `adg` | Zhang et al., *Provably Secure Generative Linguistic Steganography* (2021) |
| `discop` / `discop_base` | Ding et al., *Discop: Provably Secure Steganography in Practice Based on "Distribution Copies"* (2023) |
| `fdpss_differential_based` / `fdpss_binary_based` / `fdpss_stability_based` | Liao et al., *A Framework for Designing Provably Secure Steganography* (2025) |
| `meteor` | Kaptchuk et al., *Meteor: Cryptographically Secure Steganography for Realistic Distributions* (2021) |
| `ars` | Bai et al., *Provably Robust and Secure Steganography in Asymmetric Resource Scenario* (2025) |
| `huffman` | Yang et al., *RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks* (2019) |

```bibtex
@inproceedings{ziegler2019neural,
  title={Neural Linguistic Steganography},
  author={Zachary M. Ziegler and Yuntian Deng and Alexander M. Rush},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year={2019}
}

@inproceedings{zhang2021provably,
  title={Provably Secure Generative Linguistic Steganography},
  author={Zhang, Siyu and Yang, Zhongliang and Yang, Jinshuai and Huang, Yongfeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  year={2021}
}

@inproceedings{ding2023discop,
  title={Discop: Provably Secure Steganography in Practice Based on ``Distribution Copies''},
  author={Ding, Jinyang and Chen, Kejiang and Wang, Yaofei and Zhao, Na and Zhang, Weiming and Yu, Nenghai},
  booktitle={2023 IEEE Symposium on Security and Privacy (SP)},
  year={2023}
}

@inproceedings{liao2025fdpss,
  title={A Framework for Designing Provably Secure Steganography},
  author={Liao, Guorui and Yang, Jinshuai and Shao, Weizhi and Huang, Yongfeng},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  year={2025}
}

@inproceedings{kaptchuk2021meteor,
  title={Meteor: Cryptographically Secure Steganography for Realistic Distributions},
  author={Kaptchuk, Gabriel and Jois, Tushar M. and Green, Matthew and Rubin, Aviel D.},
  booktitle={Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security},
  year={2021}
}

@inproceedings{bai2025asymmetric,
  title={Provably Robust and Secure Steganography in Asymmetric Resource Scenario},
  author={Bai, Minhao and Yang, Jinshuai and Pang, Kaiyi and Xu, Xin and Yang, Zhen and Huang, Yongfeng},
  booktitle={2025 IEEE Symposium on Security and Privacy (SP)},
  year={2025}
}

@article{yang2019rnnstega,
  title={RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks},
  author={Yang, Zhongliang and Guo, Xiaoqing and Chen, Ziming and Huang, Yongfeng and Zhang, Yujin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2019}
}
```

## 参考资料与致谢

- 也感谢公开整理的 [Provably Secure Steganography](https://github.com/comydream/provably-secure-steganography?tab=readme-ov-file) 论文库；我们在交叉核对可证明安全隐写相关工作时，将其作为外部参考资料之一。

## License

MIT License，见 `LICENSE`。
