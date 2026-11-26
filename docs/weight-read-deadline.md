---
title: 权重读取的超时预算
---

# 权重读取的超时预算

[手册目录](index.md) · [上一篇：权重拉取失败为何会卡住训练](pull-failure.md)

把权重拉取错误传给训练器，需要先知道哪次读取已经失败。TorchTitan PR #6189 为每个生成器 rank 按读取字节量设置 deadline，同时让路由器记录失败生成器的编号和策略版本。本文核对合并提交 `7ccb1bb`。 [上游修复](https://github.com/pytorch/torchtitan/pull/6189)

## 从字节数计算等待预算

该版本在 `connect_generator` 建好读取计划后，累加 `reads` 中每项的 `nbytes`，计算 `timeout = 60 秒 + 读取字节数 / min_read_bytes_per_s`。默认速度参数为 `100e6` 字节每秒，也就是十进制 100 MB/s。这里应使用当前 rank 的计划读取量。生成器的 TP/DP 布局不同，各 rank 的读取量也可能不同。 [读取计划与 deadline 实现](https://github.com/pytorch/torchtitan/blob/7ccb1bb845ff5087bdceb1f7a7d238ba911550c9/torchtitan/rl/model_state_dict_transport.py)

| 单 rank 计划读取量 | 传输时间预算 | 加上 60 秒后的 deadline |
| --- | --- | --- |
| 2 GB，十进制 | `2e9 / 100e6 = 20 秒` | 80 秒 |
| 20 GB，十进制 | `20e9 / 100e6 = 200 秒` | 260 秒 |
| 60 GB，十进制 | `60e9 / 100e6 = 600 秒` | 660 秒 |

以前统一等待 600 秒，小分片也要等同样久。现在较小的读取有较短预算，大读取可以获得更多时间。100 MB/s 是配置中的保守速度假设，真实带宽需要按链路、并发读取和拓扑测量。 [参数说明与背景](https://github.com/pytorch/torchtitan/pull/6189)

`min_read_bytes_per_s` 在连接时用于计算一个总 deadline。代码随后把这个值传给等待全部读取完成的逻辑。因此调整它是在调整整个读取阶段的等待预算。如果想按“最近几秒是否有进展”判断卡顿，还需要相应的进度采样与判定机制。 [读取等待代码](https://github.com/pytorch/torchtitan/blob/7ccb1bb845ff5087bdceb1f7a7d238ba911550c9/torchtitan/rl/model_state_dict_transport.py)

## 失败日志与最终异常发生在不同时间

设生成器 0 先失败，生成器 1 正常加载完成，生成器 2 仍在等待。路由器会在生成器 0 失败时记录其编号和策略版本。随后用 `asyncio.gather(..., return_exceptions=True)` 等所有拉取任务结束，再收集失败索引并抛出汇总异常。如果生成器 2 后来也失败，最终错误会列出 0 和 2，并把索引最小的失败生成器所抛出的异常作为原因。异常链按任务索引选择，不保证对应时间上最早发生的失败。

因此从日志能较早找到首个失败，而控制器拿到汇总异常的时间仍受其他拉取任务影响。只观察训练器最终退出时间，会漏掉路由器已经记录的更早错误。 [并行拉取与汇总实现](https://github.com/pytorch/torchtitan/blob/7ccb1bb845ff5087bdceb1f7a7d238ba911550c9/torchtitan/rl/routing/inter_generator.py)

## 超时后检查仍在进行的读取

这一版本的文档说明，超时之后读取可能仍在进行，作业需要重启。失败生成器可能仍访问训练器的 staging buffer，因此在下一次发布重用缓冲区之前，必须先处理这些仍活跃的读取。排查记录应包含 rank、生成器编号、策略版本、计划读取字节数、配置预算、首次失败日志和最终退出时间。 [Transport 生命周期](https://github.com/pytorch/torchtitan/blob/7ccb1bb845ff5087bdceb1f7a7d238ba911550c9/torchtitan/rl/MODEL_STATE_DICT_TRANSPORT.md) · [路由器中的缓冲区说明](https://github.com/pytorch/torchtitan/blob/7ccb1bb845ff5087bdceb1f7a7d238ba911550c9/torchtitan/rl/routing/inter_generator.py)

以上 deadline 表格已用字节数复算，错误传播顺序来自公开源码。实际 RDMA 延迟、故障恢复和多卡行为待验证。
