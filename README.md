# StegoKit

`StegoKit` 是一个面向“生成式隐写算法”的模块化工具库。  
当前主接口是统一的 Stego 接口（model-in-the-loop）。

## 目录结构

```text
stegokit/
  core/          # 枚举、上下文、接口、注册表、分发器
  algo/          # 各算法实现（ac / discop / artifacts / meteor）
  utils/         # 工具（如 PRG）
  stegokit/      # 对外统一导出 API
  main.py        # 本地测试入口（示例）
```

## 安装

### 先安装 PyTorch（按你的环境选择 CPU/CUDA 版本）

`StegoKit` 不再强绑定 `torch` 版本，避免覆盖你已有环境。  
请先自行安装合适的 PyTorch，再安装 `StegoKit`。

示例（CPU）：

```bash
pip install torch
```

### 本地开发安装

```bash
pip install -e .
```

### 构建发布包

```bash
python -m build
```

构建后会在 `dist/` 下生成：

- `*.tar.gz`
- `*.whl`

## 对外 API

```python
from stegokit import (
    StegoAlgorithm,
    StegoDispatcher,
    StegoAlgorithmRegistry,
    PRG,
)
```

## 内置算法

`StegoAlgorithm` 当前包含：

- `AC`
- `DISCOP`
- `DISCOP_BASE`
- `DIFFERENTIAL_BASED`
- `BINARY_BASED`
- `STABILITY_BASED`
- `METEOR`

## 快速使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from stegokit import (
    StegoAlgorithm,
    StegoDispatcher,
    PRG,
)

model = AutoModelForCausalLM.from_pretrained("your-model-path").eval()
tokenizer = AutoTokenizer.from_pretrained("your-model-path")
dispatcher = StegoDispatcher()

prg = PRG.from_int_seed(11)
bit_stream = "010111001101011010010111"

enc = dispatcher.embed(
    algorithm=StegoAlgorithm.METEOR,
    model=model,
    tokenizer=tokenizer,
    prompt="写一段简短介绍",
    secret_bits=bit_stream,
    max_new_tokens=64,
    temperature=1.0,
    precision=16,
    prg=prg,
)

dec = dispatcher.extract(
    algorithm=StegoAlgorithm.METEOR,
    model=model,
    tokenizer=tokenizer,
    prompt="写一段简短介绍",
    generated_token_ids=enc.generated_token_ids,
    precision=16,
    prg=prg,
    max_bits=enc.consumed_bits,
)

print(enc.text)
print(enc.consumed_bits)
print(dec.bits == bit_stream[:enc.consumed_bits])
```

## 参数说明

`embed(...)` 主要参数：

- `algorithm`: 算法标识（内置可传 `StegoAlgorithm` 或对应字符串；自定义传字符串）
- `model` / `tokenizer`: 语言模型与分词器
- `prompt`: 生成上下文
- `secret_bits`: 待嵌入 bit 串
- `max_new_tokens`: 最长生成长度
- `temperature` / `top_k` / `top_p`: 采样控制
- `precision`: 精度参数
- `prg`: 随机生成器（部分算法必需）

`extract(...)` 主要参数：

- `algorithm`
- `generated_token_ids`: 已生成 token 序列
- `prompt`: 与编码时保持一致
- `precision`
- `prg`
- `max_bits`: 可选，限制最多解码 bit 数

## PRG 用法

`utils.PRG` 支持两种初始化方式：

```python
from stegokit import PRG

# 用于实验/测试
prg = PRG.from_int_seed(11)

# 用于对接配置（hex 字段）
prg = PRG.from_hex(
    input_key_hex="001122...",
    sample_seed_prefix_hex="aabbcc...",
    input_nonce_hex="ddeeff...",
)
```

说明：

- `AC` 不依赖 PRG（传了会被忽略）。
- 其余已接入算法（discop/artifacts/meteor）依赖 PRG。

## 如何注册你自己的算法

这是扩展库最重要的部分。推荐流程如下。

### 第 1 步：实现策略类

你的策略类需要实现统一接口（`encode/decode`）。  
最简单方式是按鸭子类型实现这两个方法并返回标准结果对象。

示例：

```python
from core.stego_algorithm import StegoEncodeResult, StegoDecodeResult
from core.stego_context import StegoEncodeContext, StegoDecodeContext


class MyAlgoStrategy:
    def encode(self, context: StegoEncodeContext) -> StegoEncodeResult:
        # TODO: your logic
        return StegoEncodeResult(
            generated_token_ids=[1, 2, 3],
            consumed_bits=1,
            text="demo",
            metadata={},
        )

    def decode(self, context: StegoDecodeContext) -> StegoDecodeResult:
        # TODO: your logic
        return StegoDecodeResult(bits="0", metadata={})
```

### 第 2 步：运行时注册

在你的项目启动代码中注册：

```python
from stegokit import StegoDispatcher
from algo.my_algo.my_algo import MyAlgoStrategy

dispatcher = StegoDispatcher()
dispatcher.registry.register("my_algo", MyAlgoStrategy())
```

完成后，分发器就能直接调度你的算法：

```python
dispatcher.dispatch_encode(StegoEncodeContext(algorithm="my_algo", ...))
```

说明：

- 内置算法仍建议使用 `StegoAlgorithm`，保证调用侧稳定。
- 自定义算法用字符串 key 注册与调用，避免修改 `stegokit` 包源码。
- `register("my_algo", ...)` 的 key 不能与内置算法重名（如 `ac`、`meteor`）。
