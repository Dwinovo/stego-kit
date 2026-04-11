# StegoKit CLI Plan

## 目标

为 StegoKit 增加一个可长期维护的命令行入口，让用户可以在不写 Python 代码的情况下：

- 查看当前内置算法
- 执行编码
- 执行解码

CLI 必须建立在现有核心抽象之上，而不是重新实现一套并行逻辑。也就是说，CLI 只负责：

1. 解析命令行参数
2. 读取输入文件
3. 构造 `RuntimeContext` / `GenerationConfig` / `config` / `material`
4. 调用 `StegoDispatcher`
5. 把结果写回标准输出或 JSON 文件

## 设计原则

- 薄 CLI，厚 Core：命令行层不承载算法逻辑
- 默认简单：第一次使用只需要最少参数就能跑通
- 文件优先：复杂输入尽量走 JSON 文件，不要求用户在命令行里手敲复杂结构
- 对齐现有抽象：与 `StegoDispatcher.embed/extract` 和 typed `config/material` 保持一致
- 易于扩展：后续增加新算法时，CLI 不需要大改

## 第一阶段范围

第一版 CLI 只做 3 个命令：

- `stegokit algorithms`
- `stegokit encode`
- `stegokit decode`

第一版不做：

- benchmark 子命令
- 交互式 TUI
- 批处理任务系统
- 每个算法单独一套命令树

## 用户命令设计

### 1. `stegokit algorithms`

用途：

- 列出当前内置算法
- 显示算法分类
- 显示该算法需要的 `config` 类型与 `material` 类型

建议输出字段：

- `name`
- `paradigm`
- `config_type`
- `material_type`

### 2. `stegokit encode`

用途：

- 用指定算法把 bit 串嵌入到模型生成中

建议参数：

- `--algorithm`
- `--model`
- `--messages-file`
- `--secret-bits`
- `--secret-bits-file`
- `--max-new-tokens`
- `--temperature`
- `--top-k`
- `--top-p`
- `--precision`
- `--stop-on-eos`
- `--config-json`
- `--config-file`
- `--material-json`
- `--material-file`
- `--output-file`
- `--quiet`

约束：

- `--secret-bits` 和 `--secret-bits-file` 二选一
- `--config-json` 和 `--config-file` 二选一
- `--material-json` 和 `--material-file` 二选一

### 3. `stegokit decode`

用途：

- 从已生成 token 序列中提取 bit 串

建议参数：

- `--algorithm`
- `--model`
- `--messages-file`
- `--generated-token-ids-file`
- `--max-bits`
- `--temperature`
- `--top-k`
- `--top-p`
- `--precision`
- `--config-json`
- `--config-file`
- `--material-json`
- `--material-file`
- `--output-file`
- `--quiet`

## 输入输出格式

### `messages-file`

建议使用 JSON 数组，格式直接对齐现有 `messages`：

```json
[
  {"role": "user", "content": "Write a short paragraph about privacy."}
]
```

### `generated-token-ids-file`

建议使用 JSON 数组：

```json
[101, 202, 303]
```

### `config-file`

建议使用 JSON 对象，仅包含该算法 `config` 的字段：

```json
{
  "epsilon": 0.03,
  "max_bit": 8
}
```

### `material-file`

建议使用 JSON 对象，但使用“CLI 可序列化表示”，而不是要求用户理解内部 Python 类型。

例如：

```json
{
  "prg_seed": 2026
}
```

对于第一版 CLI，建议只支持以下 material 表示：

- `NoMaterial`：不传即可
- `RandomnessMaterial`：`{"prg_seed": 2026}`
- `BitMaskMaterial`：`{"prg_seed": 2026}`

### `encode` 结果文件

建议输出 JSON：

```json
{
  "algorithm": "ac",
  "generated_token_ids": [101, 202],
  "consumed_bits": 12,
  "text": "example output",
  "encode_time_seconds": 0.42,
  "embedding_capacity": 1.5,
  "metadata": {}
}
```

### `decode` 结果文件

建议输出 JSON：

```json
{
  "algorithm": "ac",
  "bits": "010101001011",
  "decode_time_seconds": 0.18,
  "metadata": {}
}
```

## CLI 与 Core 的映射策略

CLI 不直接关心具体算法逻辑，只通过注册表读取元数据，并构造对象。

映射规则建议如下：

1. 通过 `StegoAlgorithmRegistry.default().get_spec(algorithm)` 获取算法 spec
2. 根据 spec 的 `encode_config_type` / `decode_config_type` 构造 config
3. 根据 spec 的 `encode_material_type` / `decode_material_type` 构造 material
4. 构造 `GenerationConfig`
5. 调用 `StegoDispatcher.embed/extract`

这意味着 CLI 的核心工作是“builder”：

- 从 JSON 构造 dataclass
- 从 `prg_seed` 构造 `PRG`
- 对默认 `NoConfig` / `NoMaterial` 自动兜底

## 推荐代码结构

建议新增目录：

```text
stegokit/
└── cli/
    ├── __init__.py
    ├── main.py
    ├── commands.py
    ├── builders.py
    └── io.py
```

职责建议：

- `main.py`
  - CLI 入口
  - `argparse` 定义
  - 命令分发

- `commands.py`
  - `run_algorithms`
  - `run_encode`
  - `run_decode`

- `builders.py`
  - 构造 `GenerationConfig`
  - 构造 `RuntimeContext`
  - 构造 typed `config`
  - 构造 typed `material`

- `io.py`
  - 读取 `messages.json`
  - 读取 token id 文件
  - 读取/写出 JSON 结果

## 入口配置

在 [pyproject.toml](/root/autodl-tmp/stego-kit/pyproject.toml) 中增加：

```toml
[project.scripts]
stegokit = "stegokit.cli.main:main"
```

这样安装后可直接使用：

```bash
stegokit algorithms
stegokit encode ...
stegokit decode ...
```

## 第一版实现细节建议

### 1. 命令行框架

第一版建议使用标准库 `argparse`，原因：

- 依赖少
- 稳定
- 当前命令复杂度完全足够
- 适合长期维护

后续如果 CLI 复杂度明显增加，再考虑迁移到 `Typer` 或 `Click`。

### 2. 日志与输出

建议：

- 默认命令成功时打印简洁摘要
- `--output-file` 存完整 JSON
- `--quiet` 时只输出必要结果或完全静默

### 3. 错误处理

CLI 需要把 core 抛出的异常转成用户友好的错误信息，例如：

- `messages-file` 不是合法 JSON
- `secret_bits` 含有非 `0/1`
- 指定的算法不存在
- `RandomnessMaterial` 缺少 `prg_seed`

### 4. 模型加载

第一版直接沿用 `transformers`：

- `AutoTokenizer.from_pretrained(model_path)`
- `AutoModelForCausalLM.from_pretrained(model_path)`

先不在 CLI 里做额外的设备调度抽象，只保留必要选项即可。

## 分阶段交付

### Step 1

搭建 CLI 基础骨架：

- 新增 `stegokit/cli/`
- 新增 `main.py`
- 在 `pyproject.toml` 注册 `project.scripts`

验收标准：

- `stegokit --help` 可执行
- `stegokit algorithms` 可执行

### Step 2

实现 `algorithms` 命令：

- 从注册表读取算法 spec
- 以表格或 JSON 输出算法清单

验收标准：

- 用户能看到算法名、paradigm、config 类型、material 类型

### Step 3

实现 `encode` 命令：

- 读取 messages
- 读取 secret bits
- 构造 runtime/config/material
- 调用 dispatcher
- 输出 JSON

验收标准：

- 能用 `ac` 跑通最简编码

### Step 4

实现 `decode` 命令：

- 读取 token ids
- 构造 runtime/config/material
- 调用 dispatcher
- 输出 JSON

验收标准：

- 能对 `ac` 的编码结果完成解码

### Step 5

补充算法特定配置与 material builder：

- `ADGConfig`
- `HuffmanConfig`
- `ARSEncodeConfig` / `ARSDecodeConfig`
- `RandomnessMaterial`
- `BitMaskMaterial`

验收标准：

- `adg`
- `huffman`
- `discop`
- `meteor`

至少各有一条 CLI 测试跑通

### Step 6

文档与测试补齐：

- README 增加 CLI 章节
- 增加 CLI 单元测试
- 增加一个端到端 smoke test

## 测试计划

建议新增：

- `test/test_cli_algorithms.py`
- `test/test_cli_encode_decode.py`
- `test/test_cli_builders.py`

测试重点：

- 参数解析正确
- `config/material` builder 正确
- 错误提示清晰
- 最简 `ac` encode/decode 能闭环

## 第一版使用示例

### 查看算法

```bash
stegokit algorithms
```

### 编码

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

### 解码

```bash
stegokit decode \
  --algorithm ac \
  --model /path/to/model \
  --messages-file messages.json \
  --generated-token-ids-file generated_token_ids.json \
  --max-bits 12 \
  --temperature 1.0 \
  --top-k 50 \
  --precision 16 \
  --output-file decode_result.json
```

## 非目标

以下内容不在当前计划范围内：

- 远程服务化 API
- 多任务批调度系统
- benchmark 平台
- Web UI
- 交互式对话式 CLI

## 建议结论

建议先做一个稳定的 MVP：

1. `algorithms`
2. `encode`
3. `decode`
4. JSON 文件输入输出
5. `argparse`

这样可以最快形成一个真正能用的 CLI，同时不会破坏你现在已经稳定下来的 core 架构。
