---
title: Activation offload 的带宽与时序
---

# Activation offload 的带宽与时序

[手册目录](index.md) · [相关：权重同步耗时](weight-sync.md)

训练中反向传播需要一些前向中间结果。Activation offload 把选定的 saved tensors 从 GPU 复制到 CPU，反向用到前再传回 GPU。它减少部分激活占用，同时增加 CPU 内存与链路传输。本篇核对 TorchTitan 主分支提交 `2377d43` 的实现，讨论它与 rematerialization 和 FSDP2 的组合。 [对应版本的设计文档](https://github.com/pytorch/torchtitan/blob/2377d43e20ac827dad060b45f10c3f279dd1cde5/torchtitan/distributed/ACTIVATION_OFFLOAD.md)

## 先决定保留什么，再决定放在哪里

在该实现中，remat 策略决定前向过程中哪些结果留给反向、哪些重新计算。Offloader 通过 `torch_remat.saved_tensors_hooks` 看到保留下来的张量，再按策略选择复制到 CPU。`offload_block_input` 选择 block 输入，`offload_regions` 按区域名选择保留结果。这样可以把同一块模型的重计算量、激活显存和传输量放在一起权衡。 [保留张量与策略字段](https://github.com/pytorch/torchtitan/blob/2377d43e20ac827dad060b45f10c3f279dd1cde5/torchtitan/distributed/ACTIVATION_OFFLOAD.md#what-is-offloaded)

## 参数和激活回传会争用同一链路

同时启用 FSDP2 CPU 参数 offload 时，反向边界可能要为下一个 block 做两件事。先把参数 shard 从 CPU 传到 GPU，接着执行参数 all-gather。另一路把该 block 的激活从 CPU 传回 GPU。两次 H2D 共用带宽，而参数 H2D 后面还串着 all-gather。 [FSDP2 组合说明](https://github.com/pytorch/torchtitan/pull/6137)

假设参数 H2D 独占链路需要 P 秒，激活 H2D 独占链路需要 A 秒，随后 all-gather 需要 G 秒。用一个简化模型比较两种时序，忽略其他计算、复制开销以及通信之间的争用。

| 简化时序 | 数据全部可用的时间 |
| --- | --- |
| 两次 H2D 分享带宽，到 P+A 时都完成，再做 all-gather | P+A+G |
| 参数先传，之后激活 H2D 与 all-gather 重叠 | P+max(A,G) |

例如 P=2 ms、A=3 ms、G=2 ms，第一种为 7 ms，第二种为 5 ms。这个模型能隐藏 `min(A,G)=2 ms`。如果当前 block 的反向计算已经覆盖了全部传输，这个边界的等待可能本来就很短。对真实作业，需要结合时间线与 step time 判断收益。

## 实现如何建立先后顺序

PR #6137 让 activation reload stream 等待 FSDP2 的 `all_gather_copy_in_stream`，同时保留对 compute stream 的等待。FSDP2 的 pre-backward tensor hook 先排入参数 copy-in，随后 autograd node 的 pre-hook 触发激活预取，因此新增的等待能把激活回传放到参数 copy-in 之后，让参数 all-gather 有机会与激活回传重叠。这个版本依赖 PyTorch 的 `FSDPModule.get_all_gather_copy_in_stream()` 接口。 [Reload 与 hook 实现](https://github.com/pytorch/torchtitan/blob/2377d43e20ac827dad060b45f10c3f279dd1cde5/torchtitan/distributed/activation_offload.py) · [接口依赖与测试说明](https://github.com/pytorch/torchtitan/pull/6137)

单 rank 或 `fsdp_unshard_in_current_stream` 模式中，参数 copy-in 已在 compute stream 上排队。原有的 compute-stream wait 已经保证先后顺序，新增的 copy-in-stream wait 在对应空闲流上完成。比较优化前后时，要把 FSDP 模式一起记下来。 [对应版本的 FSDP2 时序说明](https://github.com/pytorch/torchtitan/blob/2377d43e20ac827dad060b45f10c3f279dd1cde5/torchtitan/distributed/ACTIVATION_OFFLOAD.md#fsdp2)

以上耗时来自手算模型。实际实验可在同一模型和 remat 策略下记录峰值显存、CPU pinned memory、H2D/D2H 区间、all-gather 区间和训练步耗时，并检查 loss 与梯度。本项目当前完成公开源码核对与算例检查，GPU 测量待进行。
