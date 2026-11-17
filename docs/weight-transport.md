---
title: 跨布局的权重传输
---

# 跨布局的权重传输

[手册目录](index.md) · [上一篇：权重同步耗时](weight-sync.md)

训练器和生成器可以用不同的并行布局。训练器按列持有一块权重时，按行分片的生成器可能需要从多个训练 rank 各读一段。TorchTitan 主分支的可选 `ModelStateDictTransport` 在初始化时根据两侧布局规划这些读取。PR #6097 进一步把部分重排从训练端挪到了生成端。本文以合并提交 `9ea1ef0` 为准。本项目的 v0.4.0 实验仍使用 TorchStore。 [接入 PR #5997](https://github.com/pytorch/torchtitan/pull/5997) · [重排改动 #6097](https://github.com/pytorch/torchtitan/pull/6097)

## 为什么会读到需要重排的片段

设一个 4×4 权重按行存储。训练 rank A 持有前两列，rank B 持有后两列。生成 rank C 持有前两行。字母代表元素，编号代表行和列。

```text
整个权重             rank C 需要的前两行
a00 a01 a02 a03     a00 a01 a02 a03
a10 a11 a12 a13     a10 a11 a12 a13
a20 a21 a22 a23
a30 a31 a32 a33
```

从 A 读 `[a00, a01, a10, a11]`，从 B 读 `[a02, a03, a12, a13]`。这两次读取各自在源分片上连续，但 A 的元素要落到目标的偏移 `0, 1, 4, 5`，B 的元素要落到 `2, 3, 6, 7`。如果依次写入预取缓冲区，就会得到 A 的四项接 B 的四项。加载时需要把两个 2×2 片段各自写回目标矩阵的对应列。两次读取一共搬运八个元素。

## 重排放在训练端还是生成端

初版 transport 遇到这类布局时会在训练端用 DTensor redistribution 重新分片，让接收端读到可以连续写入目标张量的片段。#6097 改为训练端只做本地转换和 staging。生成端若发现某个权重的所有片段都能落在各自连续的目标区间，就按原位置预取。否则先把片段连续打包，再由 `copy_prefetched_` 把每块放回目标位置。最后同步复制流，确保下一次读取时可以安全复用缓冲区。 [合并提交中的传输实现](https://github.com/pytorch/torchtitan/blob/9ea1ef0ca8d24228500cfd6e0d635d45b57ba0f6/torchtitan/rl/model_state_dict_transport.py)

这项改动去掉了训练端 staging 的 collective，代价是部分权重需要在生成端拆包。读取字节量和读取次数是两个指标。在上游公布的单机 8×B200 微基准里，一种按第 0 维分片的生成布局读取次数从 291 增到 355，上游报告读取字节量相同。这组数据只测 transport，端到端 RL 训练要另外测。 [上游微基准与测试](https://github.com/pytorch/torchtitan/pull/6097)

以后若比较 transport，可在同一提交、同一并行布局下先记录 staging、读取和加载，再回到训练循环测实际阻塞时间与生成吞吐。这里的矩阵是手算例子，硬件数据来自上游 PR。 [训练步计时方法](weight-sync.md)
