---
title: 权重同步耗时
---

# 权重同步耗时

[手册目录](index.md) · [上一篇：吞吐测量与对照实验](throughput.md)

一次权重传输持续 3 秒，训练器可能只在某个同步点等 0.2 秒。要解释同步与异步 rollout 的性能差异，需要知道这 3 秒发生在哪一步，以及 0.2 秒落在哪条依赖上。以下以 TorchTitan v0.4.0 的 TitanRL 控制器为准。 [训练循环](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1175-L1256)

主分支随后在 PR #5235 调整了顺序，让 push 在下一步前向反向发出前完成，避免训练器的事件循环推迟 push 回复。阅读新版本日志时，应重新核对等待点与计时范围。下文的时间线对应 v0.4.0。 [后续时序调整](https://github.com/pytorch/torchtitan/pull/5235)

## 从第 N 步更新到生成器加载权重

训练器完成第 N 步的优化器更新后，控制器启动 push 和 pull 两个后台任务。Push 把训练器的 state dict 发布到 TorchStore。Pull 等 push 完成，随后要求生成器加载第 N 步权重。等所有生成器都加载完成，控制器才释放一批活跃 group 槽位，让新 group 进入。 [Push 和 pull 实现](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1265-L1292)

第 N+1 步可以在后台同步进行时等待已打包的 batch，并在 batch 就绪后做前向和反向。它在优化器改写权重前等待第 N 步的 push，在发起下一次 push 前等待第 N 步的 pull。已在生成的 group 仍可能跨越多个训练步，因此要在 batch 消费时测量策略滞后。 [等待点与后台任务](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1105-L1219) · [策略滞后](policy-lag.md)

## 日志里有两套时钟

| 指标 | 测量范围 |
| --- | --- |
| `timing/weight_sync/trainer_push_model_state_dict` | 第 N 步发起的 push 本身花了多久，由第 N+1 步记录 |
| `timing/weight_sync/generator_pull_model_state_dict` | 第 N 步发起的生成器加载花了多久，由第 N+1 步记录。计时在 push 完成后才开始，生成器加载完成就停止 |
| `timing/step/blocking_trainer_push_model_state_dict` | 第 N+1 步在优化器更新前等待 push 的时间 |
| `timing/step/blocking_generator_pull_model_state_dict` | 第 N+1 步在发起新一轮同步前等待旧 pull 任务的时间，包含等待剩余的任务收尾 |
| `perf/trainer/step_time_ratio/batch` | 第 N+1 步等训练 batch 的时间占本步总时间的比例 |

第一个训练步还没有上一轮 push 和 pull，后台耗时会记为零。分析时跳过这个启动样本，并观察预热后的连续多步。 [计时与日志](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1094-L1109) · [时间比例实现](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/metrics.py#L97-L136)

## 手算一条时间线

以第 N 步完成更新的时刻为零点。假设 push 从 0 到 1.0 秒，pull 从 1.0 到 3.0 秒，槽位随即释放。第 N+1 步在 1.8 秒走到 push 等待点，此时 push 已完成。它在 2.8 秒走到 pull 等待点，还需等 0.2 秒。此时两项后台耗时合计 3.0 秒，两项显式等待合计 0.2 秒。这个算例只说明指标之间的关系。

如果显式等待很短但 `step_time_ratio/batch` 明显上升，可以结合生成器吞吐与 group 就绪时间排查。权重加载结束前，新 group 可能还进不了缓冲区，因而一部分影响会出现在等 batch 的时间里。当前结论来自公开源码与算例，GPU 样本待采集。 [槽位释放条件](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1280-L1292)

如果训练器和生成器使用不同的分片布局，传输阶段还要决定在哪里重排张量。 [下一篇：跨布局的权重传输](weight-transport.md)
