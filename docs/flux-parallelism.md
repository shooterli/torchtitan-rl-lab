---
title: FLUX 并行配置检查
---

# FLUX 并行配置检查

[手册目录](index.md)

在 TorchTitan 中配置 `tensor_parallel_degree=2`，还需要模型声明如何把参数与激活放到 TP mesh 上。主分支 PR #5995 为 FLUX 加了更早的校验，防止未实现的并行配置进入运行阶段。本文以合并提交 `fb1d2ba` 为准。 [上游改动](https://github.com/pytorch/torchtitan/pull/5995)

## 先看配置检查会在哪里失败

| 这份实现中的设置 | 对应检查 |
| --- | --- |
| FLUX 的 TP 大于 1 | 当前布局缺少 TP 声明，完整训练配置校验抛出 `ValueError` |
| FLUX 的 EP 大于 1 | 模型没有 routed experts，校验抛出 `ValueError` |
| FLUX 的 PP 大于 1 | `FluxTrainer.Config.__post_init__` 拒绝，模型的 `pipeline()` 也有保护 |
| FLUX 的 CP 大于 1，且配置了 context load balancer | Trainer 要求使用连续分片，将 `context_parallel_load_balancer` 设为 `None` |

这些约束针对这个提交的 FLUX 实现。评估另一份 DiT 代码时，需要重新检查其布局、attention 通信与 pipeline 切分方法。 [FLUX 校验用例](https://github.com/pytorch/torchtitan/blob/fb1d2ba44a74a461dec833a7cc38d02e6309dd30/tests/unit_tests/cpu/config/test_tp_kv_heads_validation.py) · [FluxTrainer 配置](https://github.com/pytorch/torchtitan/blob/fb1d2ba44a74a461dec833a7cc38d02e6309dd30/torchtitan/models/flux/trainer.py)

## 为什么只检查整模型的轴集合会漏掉问题

假设模型有两个权重布局。第一个布局写了 TP 轴，第二个遗漏了。如果只把整模型出现过的轴合并，结果包含 TP，于是检查可能通过。#5995 改为逐个布局检查活跃的并行轴，因此可以定位到具体模块和具体布局字段。声明轴也允许显式复制，校验检查的是布局合同。实际显存与通信行为还需核对该轴的 placement 和执行路径。 [逐布局校验实现](https://github.com/pytorch/torchtitan/blob/fb1d2ba44a74a461dec833a7cc38d02e6309dd30/torchtitan/config/validation.py)

用显存算例看遗漏的影响。一个 4096×4096 权重以 BF16 存储，每个元素 2 字节，总共 32 MiB。若按输出维度平均分成两片，每个 rank 的权重数据是 16 MiB。若两边各保留完整副本，每个 rank 仍需 32 MiB。这里仅计算这个权重的数据字节，训练中还会有梯度、优化器状态、激活和临时缓冲区。先核对单个参数的本地形状，再解释完整作业的峰值显存。

## 改 degree 时重新构建模型配置

上游测试用同一个 `ParallelismConfig` 创建 FLUX 模型并调用 `validate_model_training_config`。MoE 的专家布局也在构建时决定。先按 EP=1 构建，再把训练配置改成 EP=2，可能让布局和运行设置不一致。测试新增了这个失败用例。做实验时把并行配置作为模型 builder 的输入，并保存最终配置。 [完整配置测试](https://github.com/pytorch/torchtitan/blob/fb1d2ba44a74a461dec833a7cc38d02e6309dd30/tests/unit_tests/cpu/config/test_config_validation.py)

在这个提交的完整 TorchTitan 环境中，可以先运行公开 CPU 测试，再决定是否启动多卡作业。

```bash
python -m pytest tests/unit_tests/cpu/config/test_tp_kv_heads_validation.py \
  -k TestFluxParallelismValidation -rs
```

检查通过、失败和跳过的数量。该测试类有依赖导入的跳过条件，若出现 skip，要先排查环境。本页核对了公开源码和显存算例，完整上游测试与 GPU 显存实测待进行。 [测试跳过条件](https://github.com/pytorch/torchtitan/blob/fb1d2ba44a74a461dec833a7cc38d02e6309dd30/tests/unit_tests/cpu/config/test_tp_kv_heads_validation.py)
