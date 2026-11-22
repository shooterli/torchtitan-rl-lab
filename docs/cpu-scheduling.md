---
title: CPU 调度实验
---

# CPU 调度实验

[手册目录](index.md) · [Rollout 调度](rollout-scheduling.md) · [策略滞后](policy-lag.md)

这个实验用 Python 标准库运行离散事件模拟。改变活跃槽位数和消费顺序，记录每个 group 的生成、训练时间与策略版本。模型时间由输入参数决定，程序直接跳到下一个完成事件，因此在 CPU 上可以很快复算。输出的秒数表示虚拟时间。 [模拟代码](https://github.com/shooterli/torchtitan-rl-lab/blob/main/simulate_scheduling.py)

## 模型假设

| 项目 | 默认设置 |
| --- | --- |
| 工作量 | 24 个固定 group，每个训练步消费 1 个 group，全部完成后结束 |
| 生成资源 | 固定 2 个 worker，每个同时生成 1 个 group |
| 生成时间 | 普通 group 2 秒，group 0 为慢任务，需要 12 秒 |
| 训练资源 | 单个训练器，每步 1 秒，每完成一次更新，策略版本加 1 |
| 活跃槽位 | `capacity = target + 1`，覆盖等待生成、生成中、就绪和训练中的 group |
| 权重同步 | 更新完成立即对生成器可见，训练完成才释放槽位 |
| 采样版本 | group 开始生成时记录当前版本，生成中保持该版本 |
| 策略年龄 | 消费时训练器版本减去该 group 的采样版本 |

默认从就绪的 group 中选编号最小者。`--window-groups 1` 改为严格 FIFO，等待最早尚未消费的 group。正整数窗口限制可越过慢任务的 group 编号范围。所有组使用相同的已知生成时长，两组都完成相同的工作量。

这里借用了 TitanRL 的活跃槽位和窗口概念，简化了 batch packing、权重传输、生成器内部批处理与 KV cache。每步只有一个 group，启动时可填满槽位，窗口直接以 group 数配置。缓冲更深时，两个固定 worker 可能得到更多工作，因此结果同时反映重叠与生成资源利用率。 [上游缓冲区生命周期](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/work_buffer.py) · [Windowed FIFO](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/WINDOWED_FIFO.md)

## 直接运行

在本项目根目录执行，需要 Python 3.9 或更高版本。

```bash
python3 simulate_scheduling.py
python3 simulate_scheduling.py --window-groups 1
python3 simulate_scheduling.py --slow-group -1
python3 -m unittest test_simulate_scheduling -v
```

加上 `--json` 可查看每个 group 的 `generation_start`、`generation_end`、`train_start`、`train_end`、`policy_version`、`policy_age` 和 `batch_wait`。可用 `--groups`、`--workers`、`--generation-time`、`--train-time`、`--slow-multiplier` 改动假设。`--targets 0 1 3` 指定逐组比较的缓冲深度。

## 24 个 group 的运行结果

以下是实际运行本模拟得到的虚拟时间统计，包含启动与收尾。group/s 的分子为完成的 group 数，训练器忙碌比例为总训练时间除以总历时。年龄均值按消费的 group 等权计算。

| 场景 | target | 总秒数 | group/s | 训练器忙碌 | 年龄均值 | 最大年龄 | 等 batch 秒数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 有慢任务，就绪优先 | 0 | 82 | 0.293 | 29.3% | 0.000 | 0 | 58 |
| 有慢任务，就绪优先 | 1 | 42 | 0.571 | 57.1% | 0.958 | 4 | 18 |
| 有慢任务，就绪优先 | 3 | 31 | 0.774 | 77.4% | 2.208 | 5 | 7 |
| 有慢任务，严格 FIFO | 0 | 82 | 0.293 | 29.3% | 0.000 | 0 | 58 |
| 有慢任务，严格 FIFO | 1 | 47 | 0.511 | 51.1% | 0.958 | 1 | 23 |
| 有慢任务，严格 FIFO | 3 | 36 | 0.667 | 66.7% | 2.750 | 3 | 12 |
| 等时任务，就绪优先 | 0 | 72 | 0.333 | 33.3% | 0.000 | 0 | 48 |
| 等时任务，就绪优先 | 1 | 37 | 0.649 | 64.9% | 0.958 | 1 | 13 |
| 等时任务，就绪优先 | 3 | 26 | 0.923 | 92.3% | 2.333 | 3 | 2 |

有慢任务时，target 从 0 增至 3，让更多 group 提前生成，等待从 58 秒降到 7 秒，同时最大策略年龄升至 5。target 为 3 时启用严格 FIFO，最大年龄降到 3，总历时增加 5 秒。平均年龄却从 2.208 升到 2.750，说明限制最旧样本的年龄时，其他样本也可能在队列中等得更久。 [下载全部结果 CSV](cpu-scheduling-results.csv)

## 用三个 group 手动核对

```bash
python3 simulate_scheduling.py --groups 3 --targets 1 --slow-multiplier 3 --json
python3 simulate_scheduling.py --groups 3 --targets 1 --slow-multiplier 3 --window-groups 1 --json
```

这个案例有两个槽位，group 0 生成 6 秒，其他 group 各 2 秒。

| 时刻 | 就绪优先策略发生的事件 |
| --- | --- |
| 0 | 两个 worker 开始生成 group 0 和 1，都用策略版本 0 |
| 2–3 | group 1 训练，结束后版本变为 1，释放槽位 |
| 3–5 | 空出的 worker 用版本 1 生成 group 2 |
| 5–6 | group 2 训练，结束后版本变为 2 |
| 6–7 | group 0 训练，采样版本为 0，消费时年龄为 2 |

严格 FIFO 会等 group 0 先完成，训练顺序变为 0、1、2，总历时 10 秒，最大年龄为 1。测试把这个手算时间线、串行闭式结果和 81 组资源参数组合列为核对项，检查任务完整消费、训练互斥、worker 数与活跃槽位上限、年龄和时间记账。

该实验提供一个可以修改假设的调度案例。真实 TitanRL 的 token 吞吐、显存和奖励变化需要用对应的多卡作业测量。接下来可以加入非零权重同步时间，观察槽位释放延迟如何改变等待。
