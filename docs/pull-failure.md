---
title: 权重拉取失败为何会卡住训练
---

# 权重拉取失败为何会卡住训练

[手册目录](index.md) · [权重同步的依赖](weight-sync.md) · [CPU 调度实验](cpu-scheduling.md)

同步 rollout 模式中，上一批训练完成后，要等生成器加载新权重，才能释放槽位并生成下一批。若权重拉取失败，这条路径就无法产出新 batch。TorchTitan PR #6124 修复了训练器一直等待 batch、却收不到后台拉取异常的问题。本页以合并提交 `3f09e7b` 为准。 [问题与修复](https://github.com/pytorch/torchtitan/pull/6124)

## 断开的依赖链

```text
更新参数 → 拉取新权重 → 释放活跃槽位 → 生成样本 → 产出 batch
                ↓                                    ↓
              抛出异常                          训练器等待队列
```

后台 `pull` 抛出异常后，成功路径上的槽位释放没有执行。`target_offpolicy_steps=0` 时，训练器等不到下一批。即便训练循环中还有一个 `await pull_task`，也要先越过等 batch 的位置才能执行到那里。 [修复版本的控制器](https://github.com/pytorch/torchtitan/blob/3f09e7b2b5f2b55927eb1b9275313231c94ab760/torchtitan/rl/controller/controller.py)

修复让 `training_batch_queue` 同时承载 batch、停止标记和异常。Pull 失败时，若队列有空间，立即把原异常放进去，再抛出。训练器收到异常就重新抛出，因此监督训练循环的 `run()` 能进入清理流程。如果队列已满，训练器已有一批可以消费，会在后面的 pull 等待点遇到异常。清理过程还会取消并收集未完成的控制器 pull 任务。 [控制器改动与回归用例](https://github.com/pytorch/torchtitan/pull/6124/files)

## 在 CPU 上复现等待与唤醒

在项目根目录运行下面的脚本，仅需 Python 标准库。

```bash
python3 demo_pull_failure.py
```

输出两种情况。

```text
Isolated error: pull failed, trainer is still waiting for a batch.
Forward error: trainer raised the original pull failure.
```

脚本先创建等待队列的训练任务，再让一个后台任务抛出合成的拉取错误。第一种只处理后台任务自身的异常，检查训练任务仍在等待。第二种向空队列转发原异常，检查训练任务收到相同错误。结束时两种情况都会取消并收集所有任务。这里的任务与队列用于复现异常传播，不执行模型或网络传输。 [可运行示例](https://github.com/shooterli/torchtitan-rl-lab/blob/main/demo_pull_failure.py)

## 清理控制器任务与终止远端工作

取消本地等待可以让控制器收尾，远端拉取 RPC 仍可能在 actor 销毁时运行。上游保留了这个后续问题。因此排查实际作业时，需要同时看原始拉取异常、batch 队列等待位置、控制器清理结果和远端进程状态。一个全局超时可以限制等待多久，异常传播则帮助定位哪一步先失败。 [修复中的清理说明](https://github.com/pytorch/torchtitan/pull/6124)

本页已核对公开修复，并运行 CPU 异常传播示例。完整 TitanRL、多卡通信与远端取消行为待验证。
