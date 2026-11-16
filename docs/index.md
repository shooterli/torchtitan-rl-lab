---
title: TitanRL 学习手册
---

# TitanRL 学习手册

从一个问题入手。**让生成和训练并行，能省下多少等待时间，又会让训练用到多旧的策略生成的样本？** 手册以 PyTorch TorchTitan v0.4.0 的公开源码为基准，把这两个问题拆成配置、代码和可测量的指标。 [TitanRL 上游文档](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/README.md)

| 阅读顺序 | 读完能回答的问题 |
| --- | --- |
| [Rollout 调度](rollout-scheduling.md) | 生成器和训练器如何共享样本？调大缓冲区为什么可能减少空等？ |
| [策略滞后](policy-lag.md) | 如何从权重版本算出滞后？为什么有时均值不高，个别样本却很旧？ |
| [吞吐测量与对照实验](throughput.md) | 训练 token/s 和生成 token/s 分别测什么？如何只改变调度深度？ |
| [权重同步耗时](weight-sync.md) | 一次权重传输花了多久？训练器实际等了多久？ |

每篇都可以先用纸笔算例，再跳到对应源码。项目仓库提供双组配置、固定生成并发上限的补丁和吞吐分析脚本。目前已有源码核对和合成日志验证，真实 GPU 测量待进行。 [实验代码和运行方法](https://github.com/shooterli/torchtitan-rl-lab#smoke-test)
