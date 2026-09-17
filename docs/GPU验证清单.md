# 在真实 GPU 机器上的验证清单

> 本仓的验收基线（`270 passed`、ruff 84、3 契约、7 示例、随机状态审计 `failures: 0`）
> 全部建立在 **CPU / Gloo** 上。有几类断言在那个环境里**结构上无法验证**——不是"没顾上测"，
> 而是那些代码路径**根本不执行**，或者走了另一条分支。
>
> 这份清单只列**必须换机器才能回答**的问题。每项给出：断言是什么、本机为什么测不到、
> 怎么测、**预期结果**（否则测了也不知道对不对）、以及失败意味着什么。

## 怎么用

```bash
# 0. 前置：装好 torch+CUDA，torchrun 能起 rendezvous
export PY=/path/to/gpu-venv/bin/python
export PYTHONPATH=src

# 1. 全量验收（--strict 让 skipped/blocked 也返回非零，避免静默跳过）
bash scripts/run_all_tests.sh --world-size 4 --report-dir artifacts/gpu_report --strict

# 2. 随机状态审计（多 rank，逐项打印每个 rank 的结论）
for r in 0 1; do MASTER_ADDR=127.0.0.1 MASTER_PORT=29860 WORLD_SIZE=2 RANK=$r \
  $PY scripts/verify_random_state.py & done; wait      # 看 verdict 的 failures

# 3. 并行组合矩阵（含 4/8 rank 的全轴组合）
for r in $(seq 0 3); do MASTER_ADDR=127.0.0.1 MASTER_PORT=29861 WORLD_SIZE=4 RANK=$r \
  $PY scripts/parallel_matrix.py --output artifacts/matrix.json --steps 10 & done; wait
```

排序依据是**失败的后果**，不是模块：A 类会**算错而不报错**，B 类会**报错**（容易发现），
C 类只是"有没有用"。

---

## A. 静默错误 —— 先跑这些

这几项失败时**不会抛异常**，只会让训练结果与预期不同。

### A1. 分片 checkpoint 的 CUDA 生成器

**断言**：`save_sharded` / `load_sharded` 恢复 torch-CPU、Python 与 **torch-CUDA** 三个随机流。

**本机为什么测不到**：`torch.cuda.is_available()` 恒为 False，`_rng_state()` 因此**根本不加
`"cuda"` 键**。本机只验证了"存时调了 `get_rng_state_all`、恢复时调了 `set_rng_state_all`"
这个**结构**（用假的 cuda 接口）。

**这条缺口是刚补上的**：`checkpoint/manager.py` 的 per-rank 格式一直存了 CUDA 状态，
分片格式只存了 CPU —— 也就是说在 GPU 上，从分片 checkpoint 续训的 run 会与写出它的 run
**随机流不同**，而且没有任何提示。

**怎么测**：`scripts/verify_random_state.py` 的 `checkpoint-rng` 检查已改为在
**训练设备**上取随机数（原先写死 CPU，在 GPU 上跑也测不到 CUDA 流）。在 GPU 上跑它。

**预期**：两条格式都是 `next-draw diff 0.0e+00 torch, 0.0e+00 python`。

**失败意味着**：CUDA 上的 dropout / 随机增强在续训后与写出它的 run 分岔。

### A2. GradScaler 的状态

**断言**：两种 checkpoint 格式都存/恢复 GradScaler 的状态。

**本机为什么测不到**：`torch.amp.GradScaler("cuda", enabled=True)` 在**无 CUDA 的机器上被
torch 自动降级**为 `_enabled=False`，于是 `state_dict()` **恒为 `{}`** —— 实测过。
所以两条路径在本机都是"写空、读空"，测试通过但**什么都没验证**。GPU + fp16 时
`state_dict()` 返回 5 个字段（`scale` / `growth_factor` / `backoff_factor` /
`growth_interval` / `_growth_tracker`），那条路径**才第一次真正执行**。

**怎么测**：GPU 上配 `precision.compute_dtype="float16"`（scaler 才会 enable），训练几步，
`save_sharded` → 新 trainer → `load_sharded`，比较 `trainer.scaler.state_dict()`。

**预期**：5 个字段逐个相等，尤其 `scale` 与 `_growth_tracker`。

**失败意味着**：续训时 loss scale 从初值重来——不是错结果，但多几次溢出、且与写出它的 run 不同。

### A3. Profiler 在 CUDA 上的 overlap 数字

**断言**：`overlap_ratio = hidden / (exposed + hidden)` 反映真实的通信隐藏程度。

**本机验证到哪**：Gloo 上后端在 **worker 线程**传输，所以区间真的交错，算法与 split 都验证过
（实测 2 rank：6 次通信、exposed 957/887us、hidden 35/37us、ratio 0.035/0.040）。

**GPU 上未验证**：`ProfilerActivity.CUDA` 加入后，CPU 事件与 CUDA kernel 在**同一 trace 里
的时基对齐**；以及 NCCL 的算子名是否命中 `_COLLECTIVE_BACKENDS` 的 `nccl:` 前缀。

**怎么测**：

```python
profiler = Profiler()
with profiler.capture():
    trainer.fit(batches, epochs=1)
summary = profiler.summary()
print(summary["collectives"], summary["overlap_ratio"], summary["exposed_seconds"])
```

**预期**：`collectives > 0`（若为 0，说明 `nccl:` 前缀没命中，`record_trace` 会**报错**而不是
返回 0 —— 那正是它被设计成报错的原因）；`overlap_ratio` **显著高于** CPU 上的 0.035。

**失败意味着**：要么命名匹配漏了 NCCL，要么 CPU/CUDA 时基没对齐 → 所有 overlap 结论不可信。

### A4. 激活 offload

**断言**：`offload.activation=True` 时激活被搬到 CPU，且**数值透明**。

**本机为什么测不到**：`ActivationOffloader._eligible()` 对 `device=="cpu"` 的张量返回 False
（`offload/activation.py:58`），所以本机 `offloaded` 恒为 0，整条路径没执行。

**怎么测**：GPU 上开 `offload.activation=True`，读 `trainer.offload.stats()["activation"]`；
同时用**同一 seed** 跑一遍不开 offload 的，比最终参数。

**预期**：`offloaded > 0`；两次训练的最终参数**逐位相同**。

**失败意味着**：offload 改变了数值（不该），或者根本没生效。

---

## B. 硬失败 —— 会报错，跑一次就知道

### B1. Ulysses 的多卡路径

**断言**：`UlyssesAttention` / `distributed_attention` 在多卡上数值等于 dense SDPA。

**本机为什么测不到**：`all_to_all` 没有 Gloo 实现，所以本机只验证了 world=1 的数值等价、
以及 world>1 的**拒绝**（两者都有测试）。多卡行为一行都没跑过。

**怎么测**：NCCL + world>1，把 `distributed_attention` 的输出与 dense SDPA 比。

**预期**：在容差内一致（fp32 下 ~1e-6）。

**失败意味着**：头交换的布局有错。注意 `sp_backend="ulysses"` 现在会在**校验期被拒绝**
（不是接线），所以这项要**显式调用** `UlyssesAttention`。

### B2. flash attention 的探测与 dispatch

**断言**：`flash_attention_available()` 在 CUDA + `flash_sdp_enabled()` 时为 True，且
`scaled_dot_product_attention(..., backend="flash")` 走 flash kernel 而不抛错。

**本机**：探测恒为 False。

**预期**：GPU 上为 True；`backend="flash"` 显式请求时不抛
`"flash attention execution failed without an eager fallback"`。

---

## C. 性能与收益 —— 不报错，但"有没有用"只有 GPU 能回答

### C1. 各并行组合的吞吐与显存

```bash
bash scripts/run_all_tests.sh --world-size 4 --report-dir artifacts/gpu_report
```

**预期**：`fsdp_run` 的 `parameter_max_abs_error == 0.0`（本机已是 0.0，GPU 上要确认没变）；
各组合的显存峰值随并行度下降。

### C2. `parameter_prefetch` 的逐层驱动 —— **尚未实现**

机制已修好并验证（指纹跨运行稳定、`prefetch_async` 提交即发起、预算拒绝会 fallback），
但**逐层驱动还没接**：`Trainer` 是**整模型粒度**的 `fetch`/`release`
（`trainer/step.py`），所以没有任何代码调用 `prefetch_async` 或 `finalize`。

要做它需要先解决一个设计问题：**同一批参数在反向也要用**，所以 release 的时机由**反向**的
层次结构决定，不由前向决定。另外它的收益**在 CPU 上为 0**（fetch 到 CPU 是空操作），
所以只能在这台机器上判断值不值得。

### C3. `enable_transfer_overlap` 的 H2D/D2H

`transfer_overlap` 的 scheduler 在本机 `enabled=False`（配置默认值）。GPU 上开启后
才会真的启动 bounded transfer。

### C4. mixed precision 真的降精度

`trainer/facade.py` 在非 CUDA 设备上会警告
`precision.compute_dtype='bfloat16' has no effect on device 'cpu'`。GPU 上这条警告应消失，
且 `fsdp.mixed_precision.dtype` 真的让参数以 bf16/fp16 存储（显存下降、数值变化）。

---

### C5. CUDA 上 `scaled_dot_product_attention` 是否需要本地往返

**断言（待检验）**：GPU 上 DTensor 可能**有** flash 算子的分片策略，那样
`examples/llama_shaped/model.py` 的 `_per_head` 本地往返就是多余的。

**本机为什么测不到**：CPU 上确定没有 —— 实测
`NotImplementedError: Operator aten._scaled_dot_product_flash_attention_for_cpu.default
does not have a sharding strategy registered.`。CUDA 的 flash 变体在 torch 里有注册，
但**我没在 GPU 上跑过**。

**怎么测**：GPU 上把 `_per_head` 换成直接调用 `SDPA(query, key, value)`，跑 3 步，
比较 loss。

**预期**：若 CUDA 有策略 → 去掉往返后 loss 不变，且少两次 `to_local`/`from_local`
（通信量不变，省的是打包开销）；若报同样的 `NotImplementedError` → **不能去**，
说明这是算子覆盖问题而非后端问题。

**失败意味着**：去掉后 loss 变化 → CUDA 的策略语义与"逐 head 独立"不等价，
应保留往返并记录原因。

---

## D. 本机已充分验证，GPU 上只需确认"没变"

这些**不需要重新推导**，跑一遍确认即可 —— 如果它们变了，说明 GPU 走了别的代码路径：

| 已验证的 | 本机证据 |
|---|---|
| DP+TP+PP+SP 全轴的训练与单进程一致 | world 8，`max\|diff\| = 2.98e-08` |
| 分片 checkpoint 的全部轴组合往返 | `SHARDED_AXIS_MATRIX` 覆盖 world 2/4/8 的各组合 |
| 换 world size 重分片 | tp=2 存的 checkpoint 在单进程读回，`max\|diff\| = 0.000e+00` |
| 每个 rank 只读自己那一片 | 每 rank 字节数按 `1/(dp·tp·pp)` 缩放（304 vs 2304） |
| 随机状态 5 项审计 | `init`/`rng-sync`/`sp-dropout`/`data-shard`/`checkpoint-rng`，`failures: 0` |
| overlap 算法（合成 trace 6 条） | 完全暴露/完全隐藏/部分重叠、并发 kernel 只算一次、包装层不算第二次、无通信报错 |
| 各并行组合 vs dense | `test_compounded_axes_match_dense` / `composition_train_matches_dense` |
| 死配置字段的守卫 | 39 个字段全有读取点 |

---

## 附：这份清单是怎么来的

把代码里**所有 `torch.cuda.is_available()` / `device.type == "cuda"` 的分支**列出来，
每一处就是一条"只在 GPU 上走"的路径：

```
runtime.py:38,131            设备选择与 local_rank（GPU 上才用 cuda:N）
topology.py:96               GPU 数 / 名字 / P2P 能力探测
diagnostics.py:53            报告用
profiler.py:188,250          ProfilerActivity.CUDA、显存快照   → A3
kernels/attention.py:11      flash 探测                        → B2
checkpoint/manager.py:92,244 per-rank 的 CUDA RNG（一直是对的）
trainer/checkpointing.py:87,108  sharded 的 CUDA RNG（刚补）    → A1
trainer/facade.py:96         GradScaler enable                 → A2
trainer/evaluation.py:30     归约张量放哪个设备（逻辑正确）
offload/parameter.py:173     _copy_event 的 CUDA Event 路径
offload/activation.py:58     _eligible 对 CPU 张量返回 False    → A4
```

`tests/` 里 **`cuda.is_available` 零出现** —— 测试套件没有任何 GPU 相关的 skip，
意味着上面每一条分支在 GPU 机器上都会走，而它们**都没有被专门测过**。
