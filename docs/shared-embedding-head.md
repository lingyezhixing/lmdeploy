# 共享词表与融合输出头（TurboMind 单表方案）

本文档是共享词表方案的唯一说明，内容以当前代码为准。
适用对象：使用 tied embeddings 的 LLM / Embedding 模型（TurboMind 后端；Qwen3 / Qwen3.5 系列已实测）。
文档语言为中文；代码标识符、路径与日志保持英文原文。

---

## 1. 功能概述

问题：

- tied 模型（`tie_word_embeddings: true`）的输入词表与输出头共用同一张权重，但加载器会把同一张表提交两次：`tok_embeddings` 一份、`lm_head` 一份（`lmdeploy/turbomind/models/utils.py`、`builders/_base.py`），4B 模型多耗一张表（bf16 表约 740 MiB）。
- 输出头是一次 `[tokens, hidden] x [hidden, vocab]` 的大 GEMM，显存与带宽开销大。

方案（本实现）：

- 表只加载一次，同时服务输入查表与输出头；输出头走手写 `mma.sync` 的“从词表直接出 logits”算子（`src/turbomind/kernels/logits_from_table_mma.cu`）。
- 表支持三种存储格式：`native`（保留检查点 dtype，不转换）、`int8`、`int4`（group=128）。
- 量化支持两种来源：**离线 sidecar**（推荐，多机一致、可 QA）与**加载时在线量化**（同一套数学，结果确定一致）。
- 加载路径采用 “CPU 暂存 + 提交时上传”，使加载期显存峰值 ≈ 最终模型大小。

---

## 2. 引擎接口

| 选项 | 取值 | 默认 | 含义 |
|---|---|---|---|
| `--embed-head` | `auto` / `on` / `off` | `auto` | tied 模型是否启用单表共享：`auto` 满足条件时共享、否则回退并记录原因；`on` 要求共享；`off` 走旧的双份路径 |
| `--embed-head-format` | `native` / `int8` / `int4` | `native` | 共享表的存储格式：`native` 保留检查点 dtype（bf16 就 bf16、fp16 就 fp16，无转换）；`int8`/`int4` 在加载时量化（存在 sidecar 时由 sidecar 格式覆盖，仅告警） |
| 环境变量 `LMDEPLOY_DISABLE_EMBED_QUANT=1` | — | 未设置 | 强制回退旧路径（等价于 `--embed-head off`） |

配置对象：`TurbomindEngineConfig.embed_head` / `.embed_head_format`（`lmdeploy/messages.py`），CLI 在 chat 与 serve 均可用（`lmdeploy/cli/utils.py`）。

输出头共享标志：模型构建器 `add_lm_head_shared()` 设置 `output_from_tok_embeddings=True`（`builders/text_model.py`），C++ 侧据此走融合头并跳过表的 dtype 转换。

---

## 3. 决策逻辑（`resolve_embed_head`，纯函数）

代码：`lmdeploy/turbomind/embed_quant.py`。输入包括：`tie`、sidecar 是否存在及其格式、模式、格式、环境开关、SM 版本、TP 大小、引擎 dtype、hidden 维度。

优先级（自上而下）：

1. `LMDEPLOY_DISABLE_EMBED_QUANT=1` → 旧路径（`native`）；
2. 非法模式 → 报错；
3. 非 tied 或 `--embed-head off` → 旧路径；
4. 非法 `--embed-head-format` / 非法 sidecar 格式 → 报错并列出合法值；
5. 约束检查：`SM >= 80`、`TP == 1`、引擎 dtype ∈ {fp16, bf16}、`hidden % 16 == 0`（sidecar 或量化格式还需 `hidden % 128 == 0`）；
   - 不满足时：`on` 或显式在线量化（`--embed-head-format int8/int4`）→ **响亮报错**（附约束与补救建议）；`auto` → 回退旧路径并记录一行原因（sidecar 在 `auto` 下同样回退）；
6. **存在 sidecar → 使用 sidecar**（离线优先级最高；与 `--embed-head-format` 冲突时仅告警一行，sidecar 胜出）；
7. 通过：`int8/int4` → `shared_quant`（在线量化）；`native` → `shared`（保持检查点 dtype）。

每次决策输出一行 INFO 日志，例如：

```
embed head: shared (shared native)
embed head: shared_quant (online int8)
embed head: sidecar (offline sidecar)
embed head: native (SM70 < SM80 (mma head needs SM80+))
```

`native` 下若检查点表既非 fp16 也非 bf16（如 fp32），Python 侧记录 `embed head: native table dtype torch.float32 -> engine dtype`，由 C++ 回退转换（`EnsureFloatDtype`）。

---

## 4. sidecar 契约（v2）

文件（与模型权重同目录）：

- `embed_quant.safetensors`：量化张量；
- `embed_quant.json`：元数据。

张量键（`<embed>` 为检查点里的表键，如 `model.language_model.embed_tokens.weight`）：

| 键 | dtype / 形状 | 说明 |
|---|---|---|
| `<embed>_i8` | int8 `[V, H]` | int8 对称量化表 |
| `<embed>_i8_scale` | bf16 `[V, H/128]` | int8 组 scale |
| `<embed>_i4` | uint8 `[V, H/2]` | int4 打包（低半字节=偶数列） |
| `<embed>_i4_scale` | bf16 `[V, H/128]` | int4 组 scale |
| `<embed>_i4_zero` | uint8 `[V, H/128]` | int4 组零点 |
| `lm_head.weight_from_embed` | uint8 `[1]` | 哨兵：表示用同一张表作为输出头 |

元数据字段：`version`（固定 2）、`embed_key`、`table_format`（`native`/`int8`/`int4`）、`bits`（`16/8/4`）、`group`（128）、`head_from_embed`、`created`、`qa`。

兼容性：

- 旧元数据 `table_format: bf16|fp16` 或缺失但 `bits == 16` 在读取时归一化为 `native`；
- `native` 档只写哨兵与元数据，**不复制表数据**；
- 版本不是 2 或未知格式会报错并提示用脚本重新生成；
- `--embed-head off` 或环境开关置位时 sidecar 不会被合并（`embed_quant.safetensors` 也绝不会被当作普通权重分片扫描）；
- sidecar 同样受 §3 的约束检查（`auto` 不满足时回退旧路径，`on` 报错）。

---

## 5. 离线量化脚本

```bash
python scripts/quantize_embedding.py --model <模型目录> --format native|int8|int4
```

- 默认 `native`：仅要求 tied，写哨兵-only sidecar；
- `--bits {16,8,4}` 为已弃用别名（16→native、8→int8、4→int4），与 `--format` 冲突时报错，使用时给出 DeprecationWarning；
- `--group` 只支持 128；`--chunk`（默认 8192 行）按行分块上卡量化，显存占用可控；分片 checkpoint 会自动定位包含表键的分片；
- int8 与 int4 都计算 rel-L2：超过 `--qa-threshold`（默认 int8 0.02、int4 0.12）时返回非零退出码且不写 sidecar；
- 量化在 GPU 上分块执行，产物写回 CPU/磁盘；与在线量化共用 `embed_quant.py` 中的同一套函数，结果逐位一致。

---

## 6. 加载流程与显存行为

### 6.1 时序

1. `ModelLoader.export()` → `create_checkpoint(..., embed_head=mode)`（`lmdeploy/turbomind/model_loader.py`）：
   打开 safetensors（mmap，不占显存），按需合并 sidecar（CPU）。
2. `model.model(Prefix(ckpt))` → 每个模型类先调 `add_embedding_and_head`，再加载层与 norm，最后 `builder.build()`。
3. `add_embedding_and_head`（`lmdeploy/turbomind/models/utils.py`）执行计划：
   - `sidecar` / `shared` / legacy `native`：通过 **`Prefix.get_cpu()`** 读取 CPU 张量（mmap）并直接 stage；
   - `shared_quant`：`_quantize_embed_table(table, fmt)` —— CPU 表 → `.cuda()` → **GPU 量化** → `q/scale/zero` 搬回 **CPU** → 释放 GPU 源 → `torch.cuda.empty_cache()`；
   - CPU 张量进入 builder 的 staging，直到 `build()` 才提交。
4. `build()`：`handle.param(name).alloc(...)` 分配 C++ 参数，`copy_from` 通过 `cudaMemcpyDefault` **直接从 CPU 内存 H2D 写入参数**（无中间 CUDA 临时张量），随后清空 staging。
5. C++ `ModelWeight::prepare()`（`src/turbomind/models/model_weight.cc`）：
   - int8/int4：scale（和 zero）转引擎 dtype；表本身不动；
   - **共享头且表为 fp16/bf16：跳过 `EnsureFloatDtype`**（保留检查点 dtype）；
   - 其余情况：转换为引擎 dtype（fp32 表回退路径）。

### 6.2 显存

Qwen3-Embedding-4B-AWQ（表 bf16 740.5 MiB；int8 表约 370 MiB + bf16 scale）：

| 模式 | 稳态 delta（MiB） | 加载期峰值 − 最终 |
|---|---|---|
| `off`（双份，旧路径） | 4892 | （旧路径，未优化） |
| `auto/native` | 3574 | **+3 MiB** |
| `auto/int8` | 3222 | **+0 MiB**（优化前 +1452 MiB） |

- 峰值探针：150 ms 采样 `nvidia-smi`，从进程启动到 `/health` 就绪。
- `native−int8 = 352 MiB` ≈ bf16 表与 int8 表+scale 的体积差（≈364 MiB），证明 native 稳态只持有一张表、无转换残留。

---

## 7. 运行时内核

### 7.1 输入查表（`LanguageModel::Impl::lookup`）

按表 dtype 分派（`src/turbomind/models/language_model.cc`）：

- int8 → `invokeEmbeddingLookupInt8`：`(float)q * scale`，scale dtype 必须等于输出 dtype；
- int4 → `invokeEmbeddingLookupInt4`：展开半字节后 `(q - zero) * scale`；
- 浮点表 → `invokeEmbeddingLookup`（`src/turbomind/kernels/gpt_kernels.cu`）：`<TOut, TTable>` 双模板，四种 fp16/bf16 组合，同类型位拷贝、异类型经 float 精确转换。

### 7.2 融合输出头（`invokeLogitsFromTable`）

条件：`output_from_tok_embeddings == true`（tied + 共享路径）；`PostEmbedding` 内还检查 `tp_size == 1`。

- A = 词表表 `[V, H]`（行主序，原生布局），B = 激活 x，输出 `logits [tokens, V]`；
- 三种 FORMAT：0 = 16 位浮点表（表与激活 dtype 可不同，A fragment 装载时 `cvt_pack` 转换）、1 = int8、2 = int4（量化格式在 fragment 装载时用寄存器内的 group scale（+zero）反量化）；
- mma：`mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}`，fp32 累加；
- 分块：vocab 方向 M_TILE=128（每块 256 线程）；tokens 方向 decode `N_TILE=16`、prefill `N_TILE=64`；K_STAGE=32（16 位表）/64（量化 decode），cp.async 双/三缓冲，静态 smem ≤ 48 KB；
- epilogue：fragment 先写 smem 转置 tile，再按 128 个连续 vocab 元素合并写出；
- 守卫：SM80+、`dim % 16 == 0`、表连续、量化需 `dim % 128 == 0`、logits 连续且 dtype 等于 x；不满足给出可操作错误信息；
- 旧路径（`off`/untied/不满足约束）仍用普通 GEMM 输出头，行为不变。

### 7.3 量化数学（`lmdeploy/turbomind/embed_quant.py`）

- int8 对称、group=128：`scale = max|x| / 127`（下限 1e-8），`q = round(x / scale)` 截断到 [-128,127]；scale 存 bf16；
- int4 非对称、group=128：`scale = (max - min) / 15`，`zero = round(-min / scale)`，`q = round(x/scale + zero) ∈ [0,15]`；打包 `byte = even | (odd << 4)`；scale bf16 + zero uint8；
- 在线与离线调用同一函数，保证确定性。

---

## 8. 约束与回退

| 条件 | 行为 |
|---|---|
| 非 tied / `off` / 环境开关 | 旧路径（tied 时仍是两份表） |
| sidecar 存在且满足约束 | 使用 sidecar；格式冲突仅告警 |
| `SM < 80` / `TP > 1` / 引擎 dtype 非 fp16,bf16 / `hidden % 16 != 0`（sidecar 或量化格式 + `% 128 != 0`） | `auto` 回退并记录原因；`on`/显式在线量化报错 |
| 表为 fp32 等非 16 位浮点 | 回退 C++ 引擎 dtype 转换并记录日志 |
| PyTorch 引擎 | 不涉及，行为不变 |

约束来源：融合头需要 SM80+ 的 mma；共享路径要求 TP=1；组量化要求 128 对齐。

---

## 9. 验证结果（实测）

- 性能（Qwen3.5-4B-AWQ，8 并发）：**466 tok/s**（旧双表 GEMM 头 321）；单流约 70 tok/s；长 prompt TTFT 0.64–0.79 s；int8 sidecar 路径回归与基线一致。
- native（无 sidecar）路径：8 并发约 370 tok/s、单流约 54 tok/s（表更大，带宽更高）。
- 质量：int8 teacher-forced top-1 **98.96%**；int4 QA rel-L2 9.27%（top-1 90.8%）；bf16 贪心输出与 CPU 一致；bf16→fp16 转换为位级精确（GPU 单测 `torch.equal` 验证）。
- 显存：见 §6.2；Embedding 向量 L2 范数 ≈ 1.0。
- 单测：`python -m pytest tests/turbomind/embedding/ -q` 全部通过（含决策矩阵、meta 归一化、脚本 CLI、GPU 混合精度对拍、CPU staging/峰值相关断言）。

---

## 10. 已知限制与后续项

- Embedding / Reranker 仍需生成 1 个 token 来取隐状态（`max_new_tokens=1`），未实现纯 prefill 输出（`max_new_tokens=0`）路径；
- 在线量化在量化阶段仍需源表短暂进显存（整表量化峰值约 1.1 GB，发生在层加载前，不叠加最终模型）；
- shared 头不支持 TP > 1 与 SM < 80；
- `off` 旧路径保留其固有的转换暂存开销；
- sidecar 优先级高于显式 `--embed-head-format`（冲突告警，sidecar 胜出）；
- 未支持 fp8 / fp32 表格式。

---

## 11. 代码索引

| 主题 | 位置 |
|---|---|
| 决策与量化数学、sidecar 契约 | `lmdeploy/turbomind/embed_quant.py` |
| 计划执行、CPU staging、在线量化包装 | `lmdeploy/turbomind/models/utils.py`（`add_embedding_and_head`、`_quantize_embed_table`） |
| 检查点访问（`get` / `get_cpu`）与 sidecar 合并 | `lmdeploy/turbomind/checkpoint.py` |
| 加载入口 | `lmdeploy/turbomind/model_loader.py` |
| 参数分配与提交（alloc/cast/copy_from） | `lmdeploy/turbomind/builders/_base.py`、`builders/text_model.py` |
| C++ 权重准备（dtype 规则） | `src/turbomind/models/model_weight.cc` |
| 运行时查表与融合头接线 | `src/turbomind/models/language_model.cc` |
| 融合 mma 头 | `src/turbomind/kernels/logits_from_table_mma.cu` |
| 查表内核 | `src/turbomind/kernels/gpt_kernels.cu` |
| CLI 与配置 | `lmdeploy/cli/utils.py`、`lmdeploy/messages.py` |
| 离线脚本 | `scripts/quantize_embedding.py` |
| 测试 | `tests/turbomind/embedding/` |
