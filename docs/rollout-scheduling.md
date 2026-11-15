---
title: Rollout 调度
---

# Rollout 调度

[手册目录](index.md) · [下一篇：策略滞后](policy-lag.md)

一轮在线 RL 后训练会生成样本，再用样本更新策略。TitanRL 把生成和训练放在不同的 GPU mesh 上。生成器用当前可见的权重对同一个 prompt 采样多条 completion，环境和奖励逻辑打分，批处理器把就绪的样本组打包，训练器消费 batch 并更新权重。控制器负责协调样本流和权重同步。 [架构说明](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/README.md#architecture)

```text
prompt → 生成与打分 → rollout 缓冲区 → 打包 → 训练更新
           ↑                                  │
           └──────────── 权重同步 ────────────┘
```

## 缓冲区控制了什么

把 `num_prompts_per_train_step` 记为 P，把 `target_offpolicy_steps` 记为 S。v0.4.0 允许同时活跃的 prompt group 数是 `(S + 1) × P`。一个 group 包含同一个 prompt 的多条采样。当 P 为 8 时，S 为 0 和 1 分别给出 8 和 16 个活跃 group 槽位。S 为 0 是同步基线。S 为 1 时可以让更多 group 提前生成，训练器消费当前 batch 时，生成器有机会继续为后续 batch 工作。 [控制器配置](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L150-L218)

用一个理想化算例看重叠的收益。假设每批生成耗时 4 秒，训练耗时 2 秒，忽略权重同步和排队。串行跑 3 批需要 18 秒。若生成器能持续工作，第 1、2、3 批分别在第 4、8、12 秒生成完成，训练分别在第 6、10、14 秒完成。14 秒是这个算例的结果。真实运行还取决于采样长度、批处理、KV cache 和权重同步。

## 慢样本组何时会拖住训练

设最早入队的 group 还在生成，后面几个 group 已经就绪。默认策略可以先消费较早的就绪组，让训练继续。每完成一个更新，那个慢组原先用的权重就可能再落后一步。设置 `windowed_fifo_batches` 后，较新的组只能在有限窗口内越过它。窗口被用尽时，训练器可能需要等慢组完成。 [上游 Windowed FIFO 设计](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/WINDOWED_FIFO.md)

读源码时，可以沿 `setup_async` 找到并发上限，再沿 `run` 里的 rollout workers 和 trainer loop 找到生产与消费的边界。比较实验里还要看生成器的序列上限是否跟着变化。 [控制器源文件](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py) · [吞吐测量与对照实验](throughput.md)
