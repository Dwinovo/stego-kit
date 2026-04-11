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

## 哪些算法需要什么

下面这张表是最实用的速查表。

| 算法 | 枚举值 | `config` | `material` | 适合什么时候先用 |
|---|---|---|---|---|
| Arithmetic Coding | `ac` | `NoConfig()` | `NoMaterial()` | 第一次跑通框架 |
| ADG | `adg` | `ADGConfig` | `NoMaterial()` | 想调节动态分组参数 |
| Discop | `discop` | `NoConfig()` | `RandomnessMaterial` | 需要 PRG 驱动的随机性 |
| Discop Base | `discop_base` | `NoConfig()` | `RandomnessMaterial` | Discop 基线 |
| Huffman | `huffman` | `HuffmanConfig` | `NoMaterial()` | 想控制候选集合大小 |
| Meteor | `meteor` | `NoConfig()` | `BitMaskMaterial` | 需要 bit mask |
| ARS | `ars` | `ARSEncodeConfig` / `ARSDecodeConfig` | `NoMaterial()` | 想测试 ARS 的 regular / robust 解码 |
| FDPSS Differential Based | `fdpss_differential_based` | `NoConfig()` | `RandomnessMaterial` | FDPSS 变体之一 |
| FDPSS Binary Based | `fdpss_binary_based` | `NoConfig()` | `RandomnessMaterial` | FDPSS 变体之一 |
| FDPSS Stability Based | `fdpss_stability_based` | `NoConfig()` | `RandomnessMaterial` | FDPSS 变体之一 |

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
