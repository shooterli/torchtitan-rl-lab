---
title: 吞吐测量与对照实验
---

# 吞吐测量与对照实验

[手册目录](index.md) · [上一篇：策略滞后](policy-lag.md)

| 指标 | 分子与分母 | 能回答的问题 |
| --- | --- | --- |
| `timing/step/total` | 单步墙钟时间，包含等 batch | 完成一次训练更新需要多久？ |
| `perf/trainer/tokens_per_second_full_step` | 本步全局有效训练 token 数 ÷ 单步墙钟时间 | 训练端消费样本的速度是多少？ |
| `vllm.generation_tokens` 的区间速率 | 生成 token 累计计数差 ÷ 两个采样时刻的时间差 | 生成端在这个区间产出了多少 token？ |

`rollout/output_tokens` 的均值是每轮 completion 长度。累计计数器才提供整段生成量。比如虚构的两个快照相隔 2.5 秒，计数从 48,000 增至 54,000，则速率为 2,400 token/s。计算前确认两次快照来自同一生成进程，计数未重置，两个实验的窗口都处于预热后的训练阶段。 [训练指标](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/metrics.py) · [生成计数器](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/observability/metrics/README.md)

## 固定 vLLM 的序列上限

本项目的 smoke test 每一步用 8 个 prompt group，每组采样 8 条，关闭验证，只启动一个生成器 DP rank。上游配置会根据活跃 rollout 的规模推导 `max_num_seqs`。S 从 0 改为 1 时，活跃 group 从 8 增至 16，序列上限从 64 增至 128。上限还会参与 CUDA graph 的配置，因此直接比较两组，测到的是整套默认配置的差异。 [上限推导](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L575-L635) · [生成器参数](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/generator/vllm/engine.py#L280-L313)

第二组对照把两边的上限都设为 128，再比较 S 为 0 和 1。保持模型、数据、并行拓扑、生成参数和验证设置一致。启动后先从日志确认两边生效的上限，再并排报告训练步耗时、生成 token/s、策略滞后。若吞吐变化，结合训练器的 batch 等待比例看时间具体花在哪。 [固定上限的运行方法](https://github.com/shooterli/torchtitan-rl-lab#hold-the-generator-sequence-cap-fixed)

十步 smoke test 主要验证能否启动和采到指标。性能结论要用预热后的较长窗口与重复运行，学习效果还需要独立的奖励评估。当前仓库的速率分析脚本已通过合成日志检查，等待 GPU 结果。
