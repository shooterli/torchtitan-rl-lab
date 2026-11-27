---
title: 一次在途同步的 CPU 重叠实验
---

# 一次在途同步的 CPU 重叠实验

[手册目录](index.md) · [阻塞同步基线](cpu-sync-latency.md) · [权重同步源码导读](weight-sync.md)

同步可以放到后台，训练器仍可能在参数更新前等待。这个实验让下一步计算与上一轮同步重叠，保留“最多一次同步在途”和“同步结束才释放槽位”两条限制，观察等待转移到哪里。全部结果来自 Python 标准库的 CPU 离散事件模拟，秒数是输入参数定义的虚拟时间。

## 两种调度共享的依赖

默认模式仍在每步同步完成后才开始下一步训练。加上 `--overlap-sync` 后，下一个就绪 group 可以开始计算，但必须等上一轮同步完成，才能提交自己的参数更新并启动下一轮同步。

| 事件 | 模型中的含义 |
| --- | --- |
| `train_start` | 消费 group，按此时训练器版本计算策略年龄 |
| `compute_end` | 固定时长的训练计算结束 |
| `train_end` | 参数更新完成，训练器版本加 1；可能晚于计算结束 |
| `sync_end` | 新版本对生成器可见，当前 group 的活跃槽位释放 |
| `update_wait` | `train_end − compute_end`，计算完成后等上一轮同步的时间 |

一个 group 从进入缓冲区到自己的 `sync_end` 都占用槽位。正在同步的上一组、正在计算的当前组和排队生成的组各占一个槽位。即使 worker 空闲，也不能提前重用尚未释放的槽位。相同时刻先发布旧版本，再提交当前更新，最后调度新生成；因此在发布时刻开始生成的 group 可以看到刚发布的版本。

这里把前向反向计算合成固定时长，把参数更新视为瞬时操作，把 push 和 pull 合成一个同步阶段，并要求整个上一轮同步结束后才能更新。TorchTitan v0.4.0 的实际控制器分别在优化器前等 push、在发起下一轮同步前等 pull，允许的重叠范围不同；主分支又在 #5235 调整了 push 顺序。本模型用明确的较强依赖研究一个有限重叠方案，不能直接预测任一版本的 GPU 加速比。 [v0.4.0 等待点](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1175-L1226) · [槽位释放](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1265-L1292) · [主分支时序调整](https://github.com/pytorch/torchtitan/pull/5235)

## 三组任务的手算核对

```bash
python3 simulate_scheduling.py --groups 3 --slow-group -1 --sync-time 3 --targets 1 --json
python3 simulate_scheduling.py --groups 3 --slow-group -1 --sync-time 3 --targets 1 --overlap-sync --json
```

两名生成 worker、两个活跃槽位，三个 group 的生成各需 2 秒，计算各需 1 秒，每次同步需 3 秒。

| group | 生成区间 | 计算区间 | 等待更新 | 同步区间 | 采样版本 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0–2 | 2–3 | 无 | 3–6 | 0 |
| 1 | 0–2 | 3–4 | 4–6 | 6–9 | 0 |
| 2 | 6–8 | 8–9 | 无 | 9–12 | 1 |

group 1 可以提前算完，但同步不能同时发起。group 2 要等第 6 秒释放槽位后才能生成，此时版本 1 已发布。最终从阻塞模式的 14 秒变为 12 秒，包含最后一次同步。

## 固定工作量下的对照

以下沿用 24 个 group、两个 worker、普通生成 2 秒、group 0 生成 12 秒、计算 1 秒的基础设置。每行只改变同步策略，目标缓冲深度与工作量保持一致。

| 同步秒数 | target | 阻塞总秒数 | 重叠总秒数 | 重叠 batch 等待 | 重叠更新等待 | 重叠年龄均值 | 重叠最大年龄 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 82 | 82 | 58 | 0 | 0.000 | 0 |
| 0 | 1 | 42 | 42 | 18 | 0 | 0.958 | 4 |
| 0 | 3 | 31 | 31 | 7 | 0 | 2.208 | 5 |
| 1 | 0 | 106 | 106 | 81 | 0 | 0.000 | 0 |
| 1 | 1 | 54 | 54 | 29 | 0 | 0.958 | 3 |
| 1 | 3 | 50 | 32 | 7 | 0 | 2.542 | 5 |
| 3 | 0 | 154 | 154 | 127 | 0 | 0.000 | 0 |
| 3 | 1 | 100 | 79 | 51 | 1 | 0.958 | 2 |
| 3 | 3 | 98 | 75 | 3 | 45 | 2.750 | 4 |

target 为 0 时只有一个槽位，无法同时保留同步与计算所需的两个 group，开启重叠也没有收益。同步 1 秒、target 为 1 时，总时间也保持 54 秒，说明允许重叠并不足以保证端到端加速。

同步 3 秒、target 为 3 时，batch 等待从阻塞模式的 71 秒降到 3 秒，但产生了 45 秒更新等待。总时间从 98 秒降到 75 秒；最大策略年龄从 3 升到 4。因此除了等 batch 的指标，还要保留参数更新等待和策略年龄。 [18 组原始模拟统计](cpu-sync-overlap.csv)

`batch_wait` 仍从上一步 `train_end` 算到本步 `train_start`，不包含本步计算后的等待。两种模式都满足 `总时间 = batch_wait 总和 + group 数 × 计算时长 + update_wait 总和 + 最后一次同步`。上例为 `3 + 24 × 1 + 45 + 3 = 75`。`trainer_utilization` 的分子只算有效计算时间。

## 复算与边界

```bash
python3 simulate_scheduling.py --sync-time 3 --overlap-sync
python3 sweep_sync_overlap.py > docs/cpu-sync-overlap.csv
python3 -m unittest test_simulate_scheduling -v
```

六项测试包含手算时间线、无同步和单槽位下两种模式的一致性，以及 1,458 组参数组合的事件与资源检查。检查计算互斥、同步串行、采样时可见版本、活跃槽位、worker 上限和时间记账；此前三份 CSV 的统计也已复算核对。

模型假设生成、计算和传输互不争抢带宽，生成中始终使用开始时的版本；没有模拟真实的权重快照、远端故障、KV cache 或多轮版本切换。下一步最值得拆开 push 与 pull 的耗时和等待点，再考虑资源竞争。这里的模拟结果不能替代 GPU 吞吐、显存或奖励测量。
