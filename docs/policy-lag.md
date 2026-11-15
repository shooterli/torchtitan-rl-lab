---
title: 策略滞后
---

# 策略滞后

[手册目录](index.md) · [上一篇：Rollout 调度](rollout-scheduling.md) · [下一篇：吞吐测量](throughput.md)

TitanRL 在训练器**消费 batch 时**计算策略滞后。对每个打包后的训练样本，用当前训练器的策略版本减去样本记录的最小策略版本。以更新步为单位，可以比较不同速度的两次运行。 [v0.4.0 指标实现](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/metrics.py#L170-L235)

假设消费时训练器版本为 12，四个样本各自记录的最小版本是 12、11、11、8。四个样本的滞后分别为 0、1、1、4。这个 batch 的平均滞后是 1.5 步，最大滞后是 4 步。把每步的 `train_batch/policy_age` 和 `train_batch/policy_age_max` 一起读，才能看见少数慢样本的影响。

`train_batch/pct_samples_over_target_age` 按当前运行自己的 S 判断超标。上面的样本在 S 为 0 时有 75% 超标，S 为 1 时只有 25%。即使两个实验的年龄分布完全一样，这个指标也会因为阈值不同而变化。跨组可先比较均值和最大值，拿到样本明细后再用相同阈值比较尾部比例。若各步样本数不同，跨步的样本均值要按样本数加权。 [指标计算代码](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/metrics.py#L203-L234)

## 多轮样本的版本陷阱

v0.4.0 的样本构建器会保留同一训练段开头一轮的最小版本。若开头使用版本 9，后面一轮因路由迁移用到版本 8，训练器在版本 12 消费时会报告 3 步，实际涉及的最旧版本对应 4 步。上游 PR #5954 改成对该训练段的各轮取最小值。本项目第一组实验限制每条样本为单轮。做多轮对照时，应固定包含修复的版本，并记录提交号。 [v0.4.0 样本构建器](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/trainer/training_sample_builder.py#L350-L398) · [上游修复与路由案例](https://github.com/pytorch/torchtitan/pull/5954)
