<div align="center">
  <h1>StegoKit</h1>
  <p>
    <strong>A Unified & Modular Generative Steganography Toolkit for LLMs</strong>
  </p>
  <p>
    <!-- Badges -->
    <a href="./LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8+-blue.svg">
    <img alt="Framework: PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white">
    <img alt="Library: Transformers" src="https://img.shields.io/badge/Transformers-%23FFD21E.svg?logo=HuggingFace&logoColor=black">
  </p>
</div>

---

**StegoKit** 是一个专为大语言模型（LLM）设计的生成式隐写工具包。它提供了一个统一、模块化的接口，使研究人员和开发者能够轻松地进行隐写算法的研究、交叉对比及新策略实验。

简单来说，StegoKit 主要做两件事：
- 🔒 **编码 (Encoding)**：将 `0/1` 秘密比特串无缝嵌入到 LLM 生成 token 的过程之中。
- 🔓 **解码 (Decoding)**：从生成的 token 序列中提取出隐藏的秘密比特串。

> 💡 **核心设计哲学**：本项目不仅提供特定论文的复现，更致力于构建一个**抽象的统一框架**。通过一致的 API 与 CLI 接口，你可以无缝切换不同的隐写策略并开展对比测试。

## 📖 目录

- [✨ 核心特性](#-核心特性)
- [🚀 快速开始](#-快速开始)
- [💻 CLI 命令行指南](#-cli-命令行指南)
- [🧠 架构设计：三层参数模型](#-架构设计三层参数模型)
- [📚 目前支持的算法](#-目前支持的算法)
- [🛠 自定义算法扩展](#-自定义算法扩展)
- [👨‍💻 开发者指南](#-开发者指南)
- [📜 License & 致谢](#-license--致谢)

---

## ✨ 核心特性

- **统一的调用方式**：所有算法通过相同的调度机制 (`StegoDispatcher`) 调用，接口标准一致，替换算法成本极低。
- **完备的命令行工具 (CLI)**：提供完整的 `encode` 和 `decode` 接口，无需编写代码即可快速进行测试与实验验证。
- **三层参数解耦设计**：创新性地将“通用生成参数”、“算法特有配置项”与“外部安全资源/密码学材料”分离，逻辑隔离更加清晰严谨。
- **丰富的内置算法库**：内置 Arithmetic Coding, ADG, Discop 系列, FDPSS 系列, Meteor, ARS 等十余种前沿方案的标准化实现。
- **高度的扩展能力**：依靠接口协议设计，支持用户极其容易地接入自定义隐写算法并注册到框架流中。

---

## 🚀 快速开始

如果你第一次接触 StegoKit，我们推荐使用 **Arithmetic Coding (AC)** 算法作为你的第一个例子。它无需额外的配置文件或密码材料，最容易跑通。

### 1. 安装

首先，克隆仓库并安装依赖。建议在 Python 虚拟环境中进行（`>=3.8`）：

```bash
git clone <your-repo-url>
cd stego-kit
pip install -e .
```

*(注意：如果你需要自行安装 CUDA 版本的 `torch`，请在此步骤前先完成环境搭建，再执行安装)*

### 2. 30 秒运行示例

以下是使用 Python API 快速演示编码与解码流程的最简可行示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from stegokit import StegoAlgorithm, StegoDispatcher

model_name = "<chat-model-with-chat-template>" # 例: "Qwen/Qwen2.5-0.5B-Instruct"

# 1. 加载模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 兼容不同模型结构与分词特性的必要调整
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
if not getattr(tokenizer, "chat_template", None):
    raise ValueError("StegoKit expects tokenizer.chat_template to be configured.")

# 2. 初始化调度器
dispatcher = StegoDispatcher(verbose=True)
messages = [{"role": "user", "content": "Write a short paragraph about privacy."}]
secret_bits = "010101001011"   # 将被隐藏进去的比特串

# 3. 编码嵌入 (Embedding)
enc = dispatcher.embed(
    algorithm=StegoAlgorithm.AC,
    model=model,
    tokenizer=tokenizer,
    secret_bits=secret_bits,
    messages=messages,
    max_new_tokens=64,
    temperature=1.0,
    top_k=50,
    precision=16,
)

# 4. 解码提取 (Extraction)
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
)

print(f"生成的携带秘密的文本:\n{enc.text}")
print(f"实际嵌入比特数: {enc.consumed_bits}")
print(f"解码提取比特数: {dec.bits}")
```

> ⚠️ **重要提示**：诸如 Temperature, Top-K, Precision 这样的参数，不同算法在编码和解码保持参数严格一致上的要求可能有极大区分。在实际研究运用中，请按照算法本身的论文设计以及配置提示处理。

### 3. 环境进阶：显式上下文调用

针对更严格的模型对比和架构实验任务，系统支持通过显式地封装 `RuntimeContext` 来做调度的进阶调用写法，这不仅能够彻底将环境解耦，也能完美接入算法专有的 Config 与 Material。

<details>
<summary>点击展开：进阶调用示范</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from stegokit import (
    ADGConfig, GenerationConfig, NoMaterial, RuntimeContext, 
    StegoAlgorithm, StegoDecodeContext, StegoDispatcher, StegoEncodeContext
)

model_name = "<chat-model-with-chat-template>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
dispatcher = StegoDispatcher(verbose=True)

# 1) 分离公共运行配置的 RuntimeContext
runtime = RuntimeContext(
    model=model,
    tokenizer=tokenizer,
    messages=[{"role": "user", "content": "Explain steganography briefly."}],
    generation=GenerationConfig(
        max_new_tokens=32,
        temperature=1.0,
        top_k=32,
        precision=16,
    ),
)

# 2) 进行调度：使用 ADG 算法编码
enc = dispatcher.dispatch_encode(
    StegoEncodeContext(
        algorithm=StegoAlgorithm.ADG,
        runtime=runtime,
        secret_bits="010101001011",
        config=ADGConfig(epsilon=0.03, max_bit=8),
        material=NoMaterial(),
    )
)

# 3) 进行调度：使用 ADG 算法解码
dec = dispatcher.dispatch_decode(
    StegoDecodeContext(
        algorithm=StegoAlgorithm.ADG,
        runtime=RuntimeContext(
            model=model,
            tokenizer=tokenizer,
            messages=[{"role": "user", "content": "Explain steganography briefly."}],
            generation=GenerationConfig(
                temperature=1.0, 
                top_k=32, 
                precision=16
            ),
        ),
        generated_token_ids=enc.generated_token_ids,
        max_bits=enc.consumed_bits,
        config=ADGConfig(epsilon=0.03, max_bit=8),
        material=NoMaterial(),
    )
)
```
</details>

---

## 💻 CLI 命令行指南

一旦成功安装 StegoKit 到您的环境中，您可以利用 `stegokit` 命令大幅简化生成评估。

```bash
# 获取命令帮助与支持列表
stegokit --help

# 列出当前内部已经注册支持的算法库
stegokit algorithms
```

### 快速编码 (Encode)

将对话模板存为 `messages.json`，并执行：

```bash
stegokit encode \
  --algorithm ac \
  --model /path/to/model \
  --messages-file messages.json \
  --secret-bits 010101001011 \
  --max-new-tokens 64 \
  --temperature 1.0 \
  --top-k 50 \
  --precision 16 \
  --output-file encode_result.json \
  --trust-remote-code
```

### 快速解码 (Decode)

解码基于 `encode` 生成好的结果或单 `token_ids`：

```bash
stegokit decode \
  --algorithm ac \
  --model /path/to/model \
  --messages-file messages.json \
  --generated-token-ids-file encode_result.json \
  --max-bits 12 \
  --temperature 1.0 \
  --top-k 50 \
  --precision 16 \
  --output-file decode_result.json
```

### 高级设置传参方式
应对类似具有参数旋钮或者 PRG 分离的安全隐写策略，CLI 允许在尾部接上 JSON 以指定专有字段：

```bash
# 例如，指定 ADG 的配置项 Config
--config-json '{"epsilon": 0.03, "max_bit": 8}'

# 例如，指定 Discop 的密码种子材料 Material
--material-json '{"prg_seed": 2026}'
```

---

## 🧠 架构设计：三层参数模型

要兼容种类繁多且在实施路线上各自有偏向的隐写工作，StegoKit 强迫用户将配置项分清。理解以下的 3 个分层模型结构，有助于你更好控制变量：

| 层级名称 | 承载类 | 负责的职责定义 | 思考侧重点 |
|---|---|---|---|
| **层级一 (通用)** | `GenerationConfig` | 所有算法与基座共享的模型生成超参数 (包含 `temperature`、`top_k`、`top_p`、最大长等) | "模型采样该怎么调控？" |
| **层级二 (算法)** | `Config` 等衍生物 | 针对特定的某一个算法才具有的内部逻辑调节旋钮 (如 `ADGConfig.epsilon`，`Huffman.bit_num` ) | "算法在实现过程中受何种变量制约？" |
| **层级三 (材料)** | `Material` 等衍生物 | 外部输入的密码学安全资源、比特掩码或种子源依赖 (如 `BitMaskTutorial`, `RandomnessMaterial`) | "外部应分发何种一致且安全的随机资源给通讯方？" |

---

## 📚 目前支持的算法

StegoKit 长期持续整合经典的隐写学派代表工作。目前原生覆盖与测试清单见下：

| 算法协议名称 | 系统内部枚举值 | 学术来源 / 关联论文链接 | 特殊 `config` 依赖 | 特殊 `material` 依赖 |
|---|---|---|---|---|
| **Arithmetic Coding** | `ac` | [*Neural Linguistic Steganography*](https://arxiv.org/abs/1909.01496)<br>*(EMNLP-IJCNLP 2019)* | `NoConfig()` | `NoMaterial()` |
| **ADG** | `adg` | [*Provably Secure Generative Ling...*](https://arxiv.org/abs/2106.02011)<br>*(Findings of ACL 2021)* | `ADGConfig` | `NoMaterial()` |
| **Discop** 及基础版 | `discop` <br>`discop_base` | [*Discop: Provably Secure Stega...*](https://ieeexplore.ieee.org/document/10179287)<br>*(IEEE S&P 2023)* | `NoConfig()` | `RandomnessMaterial` |
| **Huffman** | `huffman` | [*RNN-Stega: Linguistic ...*](https://ieeexplore.ieee.org/document/8470163)<br>*(IEEE TIFS 2019)* | `HuffmanConfig` | `NoMaterial()` |
| **Meteor** | `meteor` | [*Meteor: Cryptographically Secure...*](https://eprint.iacr.org/2021/686)<br>*(ACM CCS 2021)* | `NoConfig()` | `BitMaskMaterial` |
| **ARS** | `ars` | [*Provably Robust and Secure ...*](https://arxiv.org/abs/2407.13499)<br>*(IEEE S&P 2025)* | `ARSEncodeConfig` <br>`ARSDecodeConfig` | `NoMaterial()` |
| **FDPSS 各套件组** | `fdpss_differential_based`<br>`fdpss_binary_based`<br>`fdpss_stability_based` | [*A Framework for Designing Pro...*](https://www.usenix.org/conference/usenixsecurity25/presentation/liao)<br>*(USENIX Security 2025)* | `NoConfig()` | `RandomnessMaterial` |

---

## 🛠 自定义算法扩展

StegoKit 被设计为极度易扩展。如果你正在研究设计某种新策略，只需实现基础的 `StegoStrategy` 接口，然后在全局 `StegoAlgorithmRegistry` 中注册此声明，整个框架包括模型批次分发、上下文管理与 CLI 层会自动将其完全接管。

```python
from stegokit import (
    AlgorithmSpec, NoConfig, NoMaterial, StegoParadigm,
    StegoAlgorithmRegistry, StegoDispatcher, 
    StegoEncodeResult, StegoDecodeResult
)

class MyAwesomeStrategy:
    def encode(self, context) -> StegoEncodeResult:
        # 在这里实现你的定制嵌入策略逻辑...
        return StegoEncodeResult(
            generated_token_ids=[101, 202], consumed_bits=5, text="Hello", 
            metadata={"algo": "awesome"}
        )

    def decode(self, context) -> StegoDecodeResult:
        # 在这里实现你的定制提取策略逻辑...
        return StegoDecodeResult(bits="01010", metadata={"algo": "awesome"})

# 将此工作组建通过 Registry 进行框架注册
registry = StegoAlgorithmRegistry.default()
registry.register_spec(
    AlgorithmSpec(
        name="my_awesome_algo",
        paradigm=StegoParadigm.SYMMETRIC,
        strategy=MyAwesomeStrategy(),
        encode_config_type=NoConfig,
        decode_config_type=NoConfig,
        encode_material_type=NoMaterial,
        decode_material_type=NoMaterial,
    )
)

# 使用定制版的调度器即可正常调用
dispatcher = StegoDispatcher(registry=registry)
```

---

## 👨‍💻 开发者指南

我们非常欢迎开发者参与。如果你希望从源码层面熟悉本作项目骨架，推荐按照以下路线梳理核心入口：

1. [`stegokit/core/stego_dispatcher.py`](./stegokit/core/stego_dispatcher.py)：**全生命周期的核心引擎**，负责处理模型嵌入与读取请求。
2. [`stegokit/core/stego_context.py`](./stegokit/core/stego_context.py)：管理上下分发，在此可以观测到三层参数的具体拆解流通。
3. [`stegokit/core/stego_registry.py`](./stegokit/core/stego_registry.py)：框架是如何做算法解耦及名称分词映射的。
4. [`stegokit/algo/`](./stegokit/algo/)：深入了解任意内置的算法实例代码，照猫画虎即可。

> **对不同端或不熟悉 LLM 基座特性的同学预警**：
> 部分隐写算法设计之初对大模型内部的 softmax 或 float 精度差异极度敏感并可能出现误差溢出。更换部署设备及量化模式极可能导致完全正确的代码流程也面临解码位移。请保持 `temperature`/`top-k` 和运行精度（FP16/FP32/BF16）与编码端高度一致。

## 📜 License & 致谢

- **开源协议**: StegoKit 基于 [MIT License](./LICENSE) 对外开放并限制使用。  
- **特别致谢**: 本仓库在整合梳理各家模型论文协议时，广泛参考了公开的学术归档库 [Provably Secure Steganography](https://github.com/comydream/provably-secure-steganography) 列表；特别感谢社区贡献者对学术开放生态给予的持续知识支持。

## 引用 (Citation)

如果您在研究或工程项目中使用了 StegoKit，请通过以下 BibTeX 格式引用本项目：

```bibtex
@software{stegokit2026,
  author = {Wansheng Wu and Kaibo Huang and Yukun Wei and Zhongliang Yang and Linna Zhou},
  title = {StegoKit: A Unified \& Modular Generative Steganography Toolkit for LLMs},
  year = {2026},
  url = {https://github.com/your-repo-url/stego-kit}
}
```

