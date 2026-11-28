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
| [跨布局的权重传输](weight-transport.md) | 训练器和生成器的分片方向不同时，在哪里重新拼好权重？ |
| [FLUX 并行配置检查](flux-parallelism.md) | 为什么填了 TP=2，权重仍可能占满单卡？启动前该验证哪些布局？ |
| [保存可复现的运行配置](run-records.md) | 两条相同的启动命令，为什么可能跑出不同配置？该保存哪些证据？ |
| [Activation offload 的带宽与时序](activation-offload.md) | 用 CPU 内存换显存时，为什么 H2D 的先后顺序会影响等待？ |
| [截断奖励与部分得分](truncation-reward.md) | 生成被截断时，如何保留部分得分，又给未完成的回答扣分？ |
| [CPU 调度实验](cpu-scheduling.md) | 实际运行一个慢任务案例，观察缓冲深度、等待与策略年龄怎样一起变化 |
| [缓冲深度与生成并发的对照](cpu-worker-sweep.md) | 缓冲加深的收益有多少来自利用更多生成 worker？ |
| [权重拉取失败为何会卡住训练](pull-failure.md) | 后台任务已经报错，训练器为什么还在等 batch？如何传播异常？ |
| [CPU 实验中的权重同步延迟](cpu-sync-latency.md) | 延迟发布权重与释放槽位后，深缓冲还能减少多少等待？ |
| [权重读取的超时预算](weight-read-deadline.md) | 多大的权重该等多久？多个生成器失败时，什么时候报出哪一个？ |
| [一次在途同步的 CPU 重叠实验](cpu-sync-overlap.md) | 计算提前开始后，等待是否转移到参数更新？槽位够用吗？ |
| [分开 push 和 pull 后的等待](cpu-weight-stages.md) | 同样三秒同步，为何不同阶段会改变优化器的等待与重叠？ |

每篇都可以先用纸笔算例，再跳到对应源码。项目仓库提供双组配置、固定生成并发上限的补丁和吞吐分析脚本。目前已有源码核对和合成日志验证，真实 GPU 测量待进行。 [实验代码和运行方法](https://github.com/shooterli/torchtitan-rl-lab#smoke-test)
