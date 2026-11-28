---
title: 分开 push 和 pull 后，训练器在等什么
---

# 分开 push 和 pull 后，训练器在等什么

[手册目录](index.md) · [权重同步源码导读](weight-sync.md) · [上一组 CPU 重叠实验](cpu-sync-overlap.md)

同样是 3 秒权重同步，3 秒 push 与 3 秒 pull 会造成相同的等待吗？上一组模拟把它们合成一个阶段，要求参数更新等待整个同步。本实验将计算、优化器、push、pull 分开，用始终就绪的 batch 隔离控制依赖，核对等待位置和总时间。

## 三种顺序与适用版本

| 模式 | 优化器前等待什么 | 下一轮计算何时能开始 |
| --- | --- | --- |
| `combined` | 上轮 push 和 pull 都结束 | 本轮 push 发起后 |
| `split` | 上轮 push 结束；本轮 push 还要等上轮 pull | 本轮 push 发起后 |
| `inline_push` | 上轮 push 已在上轮结束前等完；本轮 push 仍要等上轮 pull | 本轮 push 完成后 |

`combined` 是一个较强依赖的对照。`split` 的等待顺序取自 v0.4.0，固定提交 `45e33cd`：先计算、等旧 push、更新参数、等旧 pull，再启动新 push 和 pull。它允许优化器与上一轮 pull 重叠。 [v0.4.0 控制器](https://github.com/pytorch/torchtitan/blob/45e33cd413b64cb37a7840cfd3c9765d3c639ae1/torchtitan/rl/controller/controller.py#L1175-L1220)

`inline_push` 对应 #5235 合并提交 `cb790dd` 的控制顺序：更新和等旧 pull 之后，同步等本轮 push 完成，再让新 pull 与下轮计算重叠。该修复处理了训练器事件循环被前向反向占用、导致 push 回复延迟的问题。本模拟假设 push 能按固定时长独立完成，没有复现这个事件循环问题，因此三种模式的数值不能用来评价该修复的实际性能。 [合并后的等待点](https://github.com/pytorch/torchtitan/blob/cb790dd08ff64c6b601a24c9730b96e2b1ca714f/torchtitan/rl/controller/controller.py#L1467-L1497) · [修复说明与实测边界](https://github.com/pytorch/torchtitan/pull/5235)

所有 batch 都假设已就绪，不模拟 rollout 生成或活跃槽位；各阶段时长固定、资源相互独立；优化器有显式耗时。起点在初始权重已加载之后，终点包含最后一次参数更新的 push 和 pull 完成。 [最后同步的等待](https://github.com/pytorch/torchtitan/blob/45e33cd413b64cb37a7840cfd3c9765d3c639ae1/torchtitan/rl/controller/controller.py#L1260-L1263)

## 三步手算：优化器可以藏在 pull 后面

```bash
python3 simulate_weight_stages.py --json
```

默认计算 1 秒、优化器 1 秒、push 1 秒、pull 2 秒，执行三步。`split` 模式的时间线如下。

| 步数 | 计算 | 优化器 | 等旧 pull | push | pull |
| --- | --- | --- | --- | --- | --- |
| 1 | 0–1 | 1–2 | 无 | 2–3 | 3–5 |
| 2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–8 |
| 3 | 5–6 | 6–7 | 7–8 | 8–9 | 9–11 |

第 2 步在第 3 秒等到旧 push 完成，就能更新参数；旧 pull 到第 5 秒才结束，其间优化器已经完成。`combined` 则到第 5 秒才开始第 2 步优化器，本轮 push 推迟到第 6 秒。三步的总时间分别为 `combined=13`、`split=11`、`inline_push=11` 虚拟秒。后两者总时间相同，等待记账却不同。

## 固定同步总时长，改变它的组成

下面每组执行 24 步，优化器固定 1 秒，`push + pull = 3 秒`。所有秒数均来自 CPU 时间线模拟，未运行 TorchTitan 或 GPU 作业。

| 计算秒数 | push 秒数 | pull 秒数 | combined 总秒数 | split 总秒数 | inline_push 总秒数 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 3 | 97 | 74 | 74 |
| 1 | 1 | 2 | 97 | 74 | 74 |
| 1 | 3 | 0 | 97 | 97 | 120 |
| 5 | 0 | 3 | 147 | 147 | 147 |
| 5 | 1 | 2 | 147 | 147 | 170 |
| 5 | 3 | 0 | 147 | 147 | 216 |

在理想 `split` 模型里，短计算、较长 pull 能覆盖优化器耗时；改成全 push 后，优化器必须等完整 push，优势消失。长计算情况下，`split` 的旧同步已被计算覆盖，放宽优化器屏障没有进一步缩短总时间。`inline_push` 不把 push 藏在下一步计算后面，因此某些输入下理想时间更长；真实实现还要考虑它解决的回复延迟，不能直接把本表的差值当作回归幅度。 [下载全部 18 组统计](cpu-weight-stages.csv)

## 用闭式结果核对程序

记步数为 `N`，计算、优化器、push、pull 时长分别为 `C、O、P、L`。第一步到权重加载完成需 `C + O + P + L`，后续每步增加以下周期：

| 模式 | 每个后续步骤增加的虚拟时间 |
| --- | --- |
| `combined` | `max(C, P + L) + O` |
| `split` | `max(C + O, P + O, P + L)` |
| `inline_push` | `max(C + O, L) + P` |

总时间为 `C + O + P + L + (N − 1) × 周期`。当 `O = 0` 时，`combined` 与 `split` 周期完全相同。因此在本模型的前提下，优化器无耗时的模拟即使拆开两段等待，也不会仅凭屏障变弱产生总时间收益；仍可以看到优化器提交时刻与等待归属发生变化。

`optimizer_wait` 是计算结束后等更新屏障的时间，`publish_wait` 是优化器结束后等旧 pull 的时间。`inline_push` 另将每步完整 push 计入 `inline_push_seconds`；其余模式的 push 在后台，最后才排空。三种模式均满足 `总时间 = N × (C + O) + optimizer_wait 总和 + publish_wait 总和 + inline_push_seconds + final_drain_seconds`。最后排空在前两种模式为 `P + L`，在 `inline_push` 为 `L`，避免把最后一次 push 算两遍。

```bash
python3 simulate_weight_stages.py --steps 24 --compute-time 5 --push-time 1 --pull-time 2
python3 simulate_weight_stages.py --optimizer-time 0
python3 sweep_weight_stages.py > docs/cpu-weight-stages.csv
python3 -m unittest test_simulate_scheduling test_simulate_weight_stages -v
```

新模型的四项测试覆盖手算时间线、729 组参数与顺序组合的闭式结果、依赖和时间记账，以及非法输入；零耗时阶段也参与验证。此前 rollout 模拟的六项测试继续通过。

本实验只研究稳定时长下的控制路径，不输出生成吞吐或策略年龄。下一步可以把 push/pull 事件接回带槽位的 rollout 模拟，再测量生成器看到新版本的时刻；真实运行则需要结合 actor 调度、内存拷贝与网络测量验证。
