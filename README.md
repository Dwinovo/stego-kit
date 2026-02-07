# stegokit

`stegokit` 是一个面向“生成式隐写算法”的模块化工具库。  
当前实现重点是：统一算法接口、统一分发器、可插拔算法注册、可复现实验（PRG 注入）。

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
    EncodeContext,
    DecodeContext,
    EncodeResult,
    DecodeResult,
    StegoDispatcher,
    AlgorithmRegistry,
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
from stegokit import StegoAlgorithm, EncodeContext, DecodeContext, StegoDispatcher, PRG

dispatcher = StegoDispatcher()

prg = PRG.from_int_seed(11)

prob_table = [0.42, 0.26, 0.18, 0.09, 0.05]
indices = [10, 20, 30, 40, 50]
bit_stream = "010111001101011010010111"

enc = dispatcher.dispatch_encode(
    EncodeContext(
        algorithm=StegoAlgorithm.METEOR,
        prob_table=prob_table,
        indices=indices,
        bit_stream=bit_stream,
        bit_index=0,
        precision=16,
        prg=prg,
    )
)

dec = dispatcher.dispatch_decode(
    DecodeContext(
        algorithm=StegoAlgorithm.METEOR,
        prob_table=prob_table,
        indices=indices,
        prev_token_id=enc.sampled_token_id,
        precision=16,
        prg=prg,
    )
)

print(enc.sampled_token_id, enc.bits_consumed)
print(dec.bits)
```

## EncodeContext / DecodeContext 说明

编码主要参数：

- `algorithm`: 算法标识（内置可传 `StegoAlgorithm` 或对应字符串；自定义传字符串）
- `prob_table`: 当前步 token 概率分布
- `indices`: 与 `prob_table` 一一对应的 token id
- `bit_stream`: 待嵌入 bit 串
- `bit_index`: 当前读取到的 bit 位置
- `precision`: 精度参数
- `prg`: 随机生成器（部分算法必需）
- `cur_interval`: 区间状态（如 AC）

解码主要参数：

- `algorithm`
- `prob_table`
- `indices`
- `prev_token_id`: 当前步生成的 token id
- `precision`
- `prg`
- `cur_interval`

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

- `AC` 不依赖 PRG（传了也会被忽略，并在分发器打印 INFO）。
- 其余已接入算法（discop/artifacts/meteor）依赖 PRG。

## 如何注册你自己的算法

这是扩展库最重要的部分。推荐流程如下。

### 第 1 步：实现策略类

你的策略类需要实现统一接口（`encode/decode`）。  
最简单方式是按鸭子类型实现这两个方法并返回标准结果对象。

示例：

```python
from core.stego_algorithm import EncodeResult, DecodeResult
from core.stego_context import EncodeContext, DecodeContext


class MyAlgoStrategy:
    def encode(self, context: EncodeContext) -> EncodeResult:
        # TODO: your logic
        return EncodeResult(
            sampled_token_id=context.indices[0],
            bits_consumed=1,
            metadata={"next_bit_index": context.bit_index},
        )

    def decode(self, context: DecodeContext) -> DecodeResult:
        # TODO: your logic
        return DecodeResult(bits="0", metadata={})
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
dispatcher.dispatch_encode(EncodeContext(algorithm="my_algo", ...))
```

说明：

- 内置算法仍建议使用 `StegoAlgorithm`，保证调用侧稳定。
- 自定义算法用字符串 key 注册与调用，避免修改 `stegokit` 包源码。
- `register("my_algo", ...)` 的 key 不能与内置算法重名（如 `ac`、`meteor`）。

