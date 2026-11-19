---
title: 保存可复现的运行配置
---

# 保存可复现的运行配置

[手册目录](index.md) · [吞吐测量与对照实验](throughput.md)

一条命令中的 recipe 名称只能告诉我们调用了哪个函数。函数默认值、依赖版本、模型权重和环境变量都会影响实际执行。比较同步与异步 rollout 时，先保存两次运行各自的配置，才能解释差异来自哪里。

## 每次启动保存解析后的配置

在 TorchTitan v0.4.0 的 recipe 中设置 `config.trainer.debug.save_config_file = "resolved_config.json"`。控制器会把配置树写入本次 `dump_folder`。本仓库的四个 recipe 已通过共同的 `_smoke` 函数开启它，每个实验目录都会有自己的配置文件。 [控制器保存逻辑](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L322-L336)

提前检查时，可以在已经准备好依赖和 checkpoint 的公开实验环境执行下列命令。`TITANRL_LAB` 指向本项目目录，当前目录应为固定版本的 TorchTitan checkout。

```bash
PYTHONPATH="$PWD:$TITANRL_LAB${PYTHONPATH:+:$PYTHONPATH}" \
  python -m torchtitan.rl.train --module smoke_recipe \
    --config sync_smoke --print-config
```

`ConfigLoader` 先调用 recipe，应用启动参数，再验证资源路径，最后才打印并退出。路径检查失败时，应先准备 checkpoint。检查 `sync_fixed_cap` 或 `async_fixed_cap` 时还要先应用仓库中的固定上限补丁，让顶层配置包含新增字段。 [配置加载顺序](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/config/loader.py#L36-L64) · [固定上限的运行方法](https://github.com/shooterli/torchtitan-rl-lab#hold-the-generator-sequence-cap-fixed)

## 对照时逐项核对什么

| 运行证据 | 本项目的核对点 |
| --- | --- |
| 上游与实验代码的提交号、补丁内容 | v0.4.0 对应的完整提交号，以及是否应用固定上限补丁 |
| 模型与数据版本 | 相同的 checkpoint revision、tokenizer 和数据来源 |
| 解析后的配置 | 相同的 batch、最大生成长度、并行拓扑、种子设置和验证设置 |
| 启动日志中的生效参数 | stock 两组的 `max_num_seqs` 分别是 64、128，fixed-cap 两组都应是 128 |
| 运行环境 | GPU 型号与数量，PyTorch、vLLM、CUDA 和关键依赖版本，显式设置的 allocator 变量 |
| 测量窗口 | 两组各自经过预热，记录采样区间、耗时和 token 计数差 |

v0.4.0 控制器会在启动时根据 buffer 规模推导生成器并发上限。这类派生数值需要和日志一起核对。完整配置、派生参数和实际硬件各提供一部分运行证据。 [并发上限推导](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L575-L609)

## 相同种子也需要重复运行

两次运行使用同一个 seed，可以减少随机初始化和采样设置的差异。异步完成次序、不同采样形状和算子实现仍可能改变结果。本实验先比较系统指标，并保留多次运行的原始区间测量。若要比较学习效果，再用固定的评估方案观察奖励及其波动。

上游在 PR #5662 修复了 allocator 默认值覆盖显式设置的问题，说明环境变量也是实验条件的一部分。跟随新提交时，应记录实际生效的 allocator 设置及其适用版本。本文核对了配置保存与打印路径，四组 recipe 的字段设置已做本地检查，CUDA 启动生成的配置文件待核对。 [Allocator 修复](https://github.com/pytorch/torchtitan/pull/5662)
