# StegoKit

StegoKit 是一个面向大语言模型（LLM）的生成式隐写工具包。它做的事情很简单：

- 编码：把 `0/1` 比特串嵌入到模型生成过程中
- 解码：从生成出的 token 序列里把比特串提取出来

这个项目的目标不是只实现某一篇论文，而是给不同隐写算法提供一套统一调用方式，方便你做实验、对比算法和接入新策略。

## 30 秒跑起来

如果你第一次接触 StegoKit，建议先用 `AC` 算法，因为它不需要额外的 `config` 或 `material`。

### 1) 安装

```bash
git clone <your-repo-url>
cd stego-kit
pip install -e .
```

如果你需要自行安装 CUDA 版本的 `torch`，先按你的环境安装好 `torch`，再执行 `pip install -e .`。

### 2) 最简可运行示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegokit import StegoAlgorithm, StegoDispatcher

model_name = "<chat-model-with-chat-template>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
if not getattr(tokenizer, "chat_template", None):
    raise ValueError("StegoKit expects tokenizer.chat_template to be configured.")

dispatcher = StegoDispatcher(verbose=True)
messages = [{"role": "user", "content": "Write a short paragraph about privacy."}]

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
)

print("generated text:", enc.text)
print("embedded bits:", enc.consumed_bits)
print("decoded bits:", dec.bits)
```

### 3) 你只需要先记住这一条

不同算法对“编码参数和解码参数是否需要保持一致”的要求并不完全相同。实际使用时，以具体算法的设计和实验设置为准。

如果你能先把上面的最简例子跑通，再回来看下面的参数说明，会轻松很多。

## CLI 使用

安装完成后，可以直接使用命令行入口：

```bash
stegokit --help
```

当前 CLI 支持 3 个命令：

- `stegokit algorithms`
- `stegokit encode`
- `stegokit decode`

### 1) 查看内置算法

```bash
stegokit algorithms
```

如果你想要机器可读的结果：

```bash
stegokit algorithms --json
```

### 2) 准备输入文件

CLI 推荐使用 JSON 文件传入 `messages`。

`messages.json` 示例：

```json
[
  {
    "role": "user",
    "content": "Write a short paragraph about privacy."
  }
]
```

### 3) 最简单的编码命令

下面这条命令对应 README 前面的 Python 最简示例：

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
  --output-file encode_result.json
```

说明：

- `--algorithm`：算法名，例如 `ac`、`adg`、`huffman`
- `--model`：Hugging Face 模型名或本地模型目录
- `--messages-file`：消息 JSON 文件
- `--secret-bits` 或 `--secret-bits-file`：二选一
- `--output-file`：把完整结果写成 JSON 文件

如果模型需要 `trust_remote_code=True`，可以再加：

```bash
--trust-remote-code
```

### 4) 最简单的解码命令

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

这里的 `--generated-token-ids-file` 可以是：

- 只包含 token id 数组的 JSON 文件
- `encode` 命令输出的完整 JSON 结果文件

例如下面两种都可以：

```json
[101, 202, 303]
```

```json
{
  "generated_token_ids": [101, 202, 303]
}
```

### 5) 带 `config` 的算法怎么传参数

像 `ADG`、`Huffman`、`ARS` 这些有专属 `config` 的算法，可以用：

- `--config-json`
- `--config-file`

例如 ADG：

```bash
stegokit encode \
  --algorithm adg \
  --model /path/to/model \
  --messages-file messages.json \
  --secret-bits 010101001011 \
  --max-new-tokens 32 \
  --temperature 1.0 \
  --top-k 32 \
  --precision 16 \
  --config-json '{"epsilon": 0.03, "max_bit": 8}'
```

例如 Huffman：

```bash
--config-json '{"bit_num": 3}'
```

### 6) 带 `material` 的算法怎么传参数

像 `Discop`、`FDPSS`、`Meteor` 这些算法需要额外 `material`，可以用：

- `--material-json`
- `--material-file`

目前 CLI 对这类 `material` 的推荐写法是传 `prg_seed`，例如：

```bash
stegokit encode \
  --algorithm discop \
  --model /path/to/model \
  --messages-file messages.json \
  --secret-bits 010101001011 \
  --max-new-tokens 32 \
  --temperature 1.0 \
  --top-k 32 \
  --precision 16 \
  --material-json '{"prg_seed": 2026}'
```

`Meteor` 也同样可以传：

```bash
--material-json '{"prg_seed": 2026}'
```

### 7) 常用补充参数

- `--quiet`：不把 JSON 结果打印到标准输出
- `--output-file`：把完整 JSON 结果写到文件
- `--top-p`：如果你需要 nucleus sampling，可以传它
- `--stop-on-eos`：仅 `encode` 命令可用

## 三层参数模型

StegoKit 现在把参数分成 3 层。理解这 3 层，就基本理解了整个框架。

| 层 | 作用 | 典型内容 |
|---|---|---|
| `GenerationConfig` | 所有算法共享的生成参数 | `temperature`、`top_k`、`top_p`、`precision`、`stop_on_eos`、`max_new_tokens` |
| `config` | 某个算法自己的逻辑参数 | 例如 ADG 的 `epsilon`，Huffman 的 `bit_num` |
| `material` | 算法依赖的外部安全资源 | 例如 `PRG`、bit mask |

可以把它们理解成 3 个问题：

1. 模型应该怎么生成？
答案在 `GenerationConfig`。

2. 这个算法内部该怎么工作？
答案在算法自己的 `config`。

3. 这个算法运行时还需要什么额外资源？
答案在 `material`。

### 1) `GenerationConfig` 是什么

`GenerationConfig` 是所有算法共用的生成控制参数，定义在 [generation_config.py](/root/autodl-tmp/stego-kit/stegokit/core/generation_config.py)。

它包含：

- `temperature`
- `top_k`
- `top_p`
- `precision`
- `stop_on_eos`
- `max_new_tokens`

这些参数不属于某一个具体算法，而是“模型在生成时怎么采样”的通用设置。

一句话记忆：

- 只要是所有算法都会关心的生成参数，就放 `GenerationConfig`

### 2) `config` 是什么

`config` 是某个算法自己的参数，也就是“这个算法的旋钮”。

例如：

- `ADGConfig(epsilon=0.03, max_bit=8)`
- `HuffmanConfig(bit_num=3)`
- `ARSDecodeConfig(decode_mode="robust", robust_search_window=1000)`

一句话记忆：

- 如果这个参数只对某一个算法有意义，就放它自己的 `config`

### 3) `material` 是什么

`material` 是算法运行时依赖的外部资源，定义在 [security_material.py](/root/autodl-tmp/stego-kit/stegokit/core/security_material.py)。

当前内置算法真正会用到的 `material` 只有三种：

- `NoMaterial()`：不需要额外资源
- `RandomnessMaterial(prg=PRG.from_int_seed(...))`：需要 `generate_random`
- `BitMaskMaterial(prg=PRG.from_int_seed(...))`：需要 `generate_bits`

一句话记忆：

- 如果它更像“资源对象”而不是“算法旋钮”，就放到 `material`

## 目前支持的算法

下面这张表是最实用的速查表。

| 算法 | 枚举值 | 论文 | `config` | `material` |
|---|---|---|---|---|
| Arithmetic Coding | `ac` | [Neural Linguistic Steganography, EMNLP-IJCNLP 2019](https://arxiv.org/abs/1909.01496) | `NoConfig()` | `NoMaterial()` |
| ADG | `adg` | [Provably Secure Generative Linguistic Steganography, Findings of ACL-IJCNLP 2021](https://arxiv.org/abs/2106.02011) | `ADGConfig` | `NoMaterial()` |
| Discop | `discop` | [Discop: Provably Secure Steganography in Practice Based on "Distribution Copies", IEEE S&P 2023](https://ieeexplore.ieee.org/document/10179287) | `NoConfig()` | `RandomnessMaterial` |
| Discop Base | `discop_base` | [Discop: Provably Secure Steganography in Practice Based on "Distribution Copies", IEEE S&P 2023](https://ieeexplore.ieee.org/document/10179287) | `NoConfig()` | `RandomnessMaterial` |
| Huffman | `huffman` | [RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks, IEEE TIFS 2019](https://ieeexplore.ieee.org/document/8470163) | `HuffmanConfig` | `NoMaterial()` |
| Meteor | `meteor` | [Meteor: Cryptographically Secure Steganography for Realistic Distributions, ACM CCS 2021](https://eprint.iacr.org/2021/686) | `NoConfig()` | `BitMaskMaterial` |
| ARS | `ars` | [Provably Robust and Secure Steganography in Asymmetric Resource Scenario, IEEE S&P 2025](https://arxiv.org/abs/2407.13499) | `ARSEncodeConfig` / `ARSDecodeConfig` | `NoMaterial()` |
| FDPSS Differential Based | `fdpss_differential_based` | [A Framework for Designing Provably Secure Steganography, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/liao) | `NoConfig()` | `RandomnessMaterial` |
| FDPSS Binary Based | `fdpss_binary_based` | [A Framework for Designing Provably Secure Steganography, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/liao) | `NoConfig()` | `RandomnessMaterial` |
| FDPSS Stability Based | `fdpss_stability_based` | [A Framework for Designing Provably Secure Steganography, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/liao) | `NoConfig()` | `RandomnessMaterial` |

说明：FDPSS 系列当前只接受这 3 个名称：

- `fdpss_differential_based`
- `fdpss_binary_based`
- `fdpss_stability_based`

旧名称 `differential_based` / `binary_based` / `stability_based` 已移除。

## 从最简调用到进阶调用

### 最简单的调用方式：`embed` / `extract`

如果你只是想先跑通，用：

- `dispatcher.embed(...)`
- `dispatcher.extract(...)`

这两个便捷方法就够了。它们会自动帮你构造 `RuntimeContext`，也会在需要时自动补默认的 `NoConfig()` / `NoMaterial()`。

### 进阶调用方式：显式构造 `RuntimeContext`

如果你想把“运行时参数”和“算法参数”彻底分开，推荐用 `RuntimeContext`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegokit import (
    ADGConfig,
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

dispatcher = StegoDispatcher(verbose=True)
messages = [{"role": "user", "content": "Explain steganography briefly."}]

runtime = RuntimeContext(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    generation=GenerationConfig(
        max_new_tokens=32,
        temperature=1.0,
        top_k=32,
        precision=16,
    ),
)

enc = dispatcher.dispatch_encode(
    StegoEncodeContext(
        algorithm=StegoAlgorithm.ADG,
        runtime=runtime,
        secret_bits="010101001011",
        config=ADGConfig(epsilon=0.03, max_bit=8),
        material=NoMaterial(),
    )
)

dec = dispatcher.dispatch_decode(
    StegoDecodeContext(
        algorithm=StegoAlgorithm.ADG,
        runtime=RuntimeContext(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            generation=GenerationConfig(
                temperature=1.0,
                top_k=32,
                precision=16,
            ),
        ),
        generated_token_ids=enc.generated_token_ids,
        max_bits=enc.consumed_bits,
        config=ADGConfig(epsilon=0.03, max_bit=8),
        material=NoMaterial(),
    )
)
```

### 什么时候需要 `RandomnessMaterial`

`Discop` 和 `FDPSS` 系列需要 `RandomnessMaterial`。最常见的写法是：

```python
from stegokit import NoConfig, PRG, RandomnessMaterial

material = RandomnessMaterial(prg=PRG.from_int_seed(2026))
config = NoConfig()
```

编码和解码时要使用同样的 PRG 初始化参数，否则无法正确恢复比特串。

### 什么时候需要 `BitMaskMaterial`

`Meteor` 需要 `BitMaskMaterial`：

```python
from stegokit import BitMaskMaterial, NoConfig, PRG

config = NoConfig()
material = BitMaskMaterial(prg=PRG.from_int_seed(2026))
```

## 重要参数说明

### `RuntimeContext`

`RuntimeContext` 定义在 [runtime_context.py](/root/autodl-tmp/stego-kit/stegokit/core/runtime_context.py)，包含：

- `model`
- `tokenizer`
- `messages`
- `generation`

其中：

- `model` / `tokenizer` 应该来自 Hugging Face `transformers`
- `tokenizer` 需要支持 `apply_chat_template`
- `messages` 必须是对话消息列表，每条消息至少要有 `role`，并包含 `content` 或 `tool_calls`

### `ADGConfig`

定义在 [algorithm_config.py](/root/autodl-tmp/stego-kit/stegokit/core/algorithm_config.py)。

- `epsilon`：判断动态分组是否足够接近 `0.5 / 0.5`
- `max_bit`：单个 token 最多消费多少 bit

### `HuffmanConfig`

定义在 [algorithm_config.py](/root/autodl-tmp/stego-kit/stegokit/core/algorithm_config.py)。

- `bit_num`：候选大小为 `2 ** bit_num`
- `candidate_count`：直接指定候选数量；如果传了它，就优先使用它

### `ARSEncodeConfig` / `ARSDecodeConfig`

定义在 [algorithm_config.py](/root/autodl-tmp/stego-kit/stegokit/core/algorithm_config.py)。

编码侧：

- `seed`
- `secure_parameter`
- `func_type`

解码侧额外增加：

- `decode_mode`：`regular` 或 `robust`
- `robust_search_window`

说明：当前 ARS 实现不需要额外 `material`，只需要它自己的 `config`。

## 结果对象

编码返回 `StegoEncodeResult`：

- `generated_token_ids`：生成出的 token id 序列
- `consumed_bits`：实际嵌入的 bit 数
- `text`：由 `generated_token_ids` 解码得到的文本
- `encode_time_seconds`：编码耗时
- `embedding_capacity`：平均每个 token 嵌入多少 bit
- `metadata`：算法附加信息

解码返回 `StegoDecodeResult`：

- `bits`：提取出的 bit 串
- `decode_time_seconds`：解码耗时
- `metadata`：算法附加信息

## 自定义算法扩展

如果你要接入自己的算法，实现 `StegoStrategy` 协议，然后注册一个 `AlgorithmSpec` 即可。

```python
from stegokit import (
    AlgorithmSpec,
    NoConfig,
    NoMaterial,
    StegoAlgorithmRegistry,
    StegoDecodeResult,
    StegoDispatcher,
    StegoEncodeResult,
    StegoParadigm,
)


class MyStrategy:
    def encode(self, context):
        return StegoEncodeResult(
            generated_token_ids=[],
            consumed_bits=0,
            text="",
            metadata={"name": "my_algo"},
        )

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
```

## 使用注意事项

- `secret_bits` 必须是只包含 `0` 和 `1` 的字符串
- 编码和解码的参数约束取决于具体算法，不要默认所有算法都要求完全相同的设置
- 某些算法对 `temperature` / `top_k` / `top_p` / `precision` 很敏感，稍微变动就可能导致无法正确解码
- 当前项目更偏研究和实验用途，建议你在自己的数据和模型上做单独评估

## 开发者说明

项目对外 API 主要在：

- [stegokit/__init__.py](/root/autodl-tmp/stego-kit/stegokit/__init__.py)
- [stegokit/core](/root/autodl-tmp/stego-kit/stegokit/core)
- [stegokit/algo](/root/autodl-tmp/stego-kit/stegokit/algo)

如果你是第一次看代码，可以优先按这个顺序读：

1. [stego_dispatcher.py](/root/autodl-tmp/stego-kit/stegokit/core/stego_dispatcher.py)
2. [stego_context.py](/root/autodl-tmp/stego-kit/stegokit/core/stego_context.py)
3. [stego_registry.py](/root/autodl-tmp/stego-kit/stegokit/core/stego_registry.py)
4. 具体算法实现目录

## 参考资料与致谢

- 也感谢公开整理的 [Provably Secure Steganography](https://github.com/comydream/provably-secure-steganography?tab=readme-ov-file) 论文库；我们在交叉核对可证明安全隐写相关工作时，将其作为外部参考资料之一。

## License

MIT License，见 `LICENSE`。
