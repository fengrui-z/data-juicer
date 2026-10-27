# Compute Flow 首版验收（2026-10-27）

**结论：固定 Ray 常驻池上的全局调度与 Actor 内局部 batch 双层架构已实现、通过
回归和 8 卡真实推理验收；尚未证明它优于现有共享池调度或共置并行执行组。**

## 分支与提交

- 分支：`feat/elasticjuicer-compute-flow`。
- 基础：Resident DAG v0.2.0，`77885fb1333e6196deba143f0e0db9abc2351fdd`。
- `943dcf4`：ComputeFlowController、执行器接入、独立 batch 起点、控制器与 Ray 回归。
- `6e8886a`：可独立运行的真实图片算子、实验脚本和方案。
- `545cd40`：按真实图片尺寸在建池前推导解码容量；五组主验收使用此提交。
- `73974b3`：增加共置并行执行组对照；补充组使用此提交。只扩展实验分组与文档，
  产品执行器、控制器和模型算子源码与五组主验收一致，交叉审计已核对 SHA256。

本地没有 push、创建线上 PR 或生成 ZIP。依用户要求，ZIP 在确认后制作。
此分支依赖 Resident DAG 及其 ElasticJuicer 前置代码；提交 DJ 主库时需要明确
前置 PR/基线，不能把这几个增量提交描述成独立适用于主库的补丁。

## 验证范围

115 项回归全部通过（包含 24 项新增测试）。覆盖规划预算、成本变化后配额反转、
周期与迟滞、饥饿保护、少卡多组、空闲借用、静态 slot 所有权、各池计划版本、
真实 Ray 分支/汇合、模型复用、局部 OOM 学习隔离和失败关闭。
回归中的 OOM 是显式 CPU fixture，不是物理显存压力性能证据。

真实环境：8 × NVIDIA H20，40 个声明 Ray CPU，512 张 SHA256 唯一 Qwen 原图。
CPU 解码/校验 → GPU QualiCLIP 与 DINOv2-large 分支 → CPU 汇合/特征检查 → 完整 NPZ 导出。
前 256 张使用 DINO 448，后 256 张使用 896；QualiCLIP 始终 224。每组全新建池、
连续 2 pass，同池第二遍复用模型和控制状态。六组共 6,144 条已完成推理记录，
仍只有 512 张独立图片。

外层 batch 32，GPU microbatch 上限 32，自适应初值 8。每个 GPU Actor 声明
1 GPU/4 CPU，CPU 池为 4 Actor × 2 CPU；总量相同。五个调度/batch 对照使用
同一物理计划，第六组仅合并两条 GPU 分支为一个 `parallel=True` 的执行组。
它使用 Resident DAG 的执行组能力，不是 DJ 原生 FusedSequentialBatchOp。

源图最大为 5792 × 8688；保守的解码批占用估计最大 653.59 MiB，因此本轮每批
上限取 704 MiB，估计对象预算 64 GiB，有效在途上限 16。该预算是保守预约，
不是测得的进程峰值内存。启动完成时各 GPU Actor 的 PyTorch 已分配显存约
0.951 GiB，只代表已加载模型/张量，不包含推理峰值和全部 CUDA 上下文成本。

## 耗时

单位：秒。冷作业包括建池、模型加载、首次处理和完整导出；warm 是同池第二遍
处理与完整导出。两列都不含 Ray 启动和事后数值审计。文件缓存未清空。

| 对照 | 建池至首次导出 | 同池第二遍 |
|---|---:|---:|
| legacy-fixed：原共享池策略 | 29.45 | 14.02 |
| static-fixed：固定 4/4，固定 batch | 32.73 | 17.57 |
| flow-fixed：全局动态，固定 batch | 30.54 | 15.25 |
| static-adaptive：固定 4/4，局部自适应 | 33.45 | 17.51 |
| flow-adaptive：全局动态＋局部自适应 | 31.13 | 15.36 |
| grouped-fixed：共置并行执行组 | 29.43 | 14.40 |

按这一次 warm 测量计算，全局层相对固定 4/4＋固定 batch 耗时少 13.2%；双层
相对固定 4/4＋自适应 batch 少 12.3%。但双层相对原共享池多 9.6%，相对共置
并行组多 6.7%。每组只有一次 warm 测量，**没有统计显著性或规模化提速结论**。
共置组为后补控制，并非预先轮换的完整六组配对块。

原 v0.2 已允许共享池 Actor 接不同组任务；本次新增的是基于积压和服务速率的
目标分配、等待保护、控制周期与可审计租约。旧共享池本身已有较强的算力复用，
所以固定 4/4 不是最强基线，也不能只凭胜过 4/4 就宣称新调度总体更快。
无物理 OOM 时，小 batch 起步的额外探索也不保证提速；本轮自适应控制值逐步
增长，实际成功切片包含尾批，不能把这些切片大小都当成新的目标 batch。

## 机制与输出证据

- 六组均实际使用 8 张不同 GPU；每个 Actor 同一算子的实例 ID 跨两个 pass 不变。
- flow-fixed 的 8 个 GPU Actor 均执行过两个组，累计 32 次换组；flow-adaptive
  也是全部 8 个 Actor，累计 30 次换组。静态两组保持 slot 归属。
- 动态目标确实变化，包含 `quality=1 / embedding=7`，以及质量阶段清空后的
  `quality=0 / embedding=8`；没有超出总池预算，也没有跨任务重叠租用同一 Actor。
- 每组都存在同一 batch 的真实 GPU 分支重叠；两种动态模式均为 30/32 个 fork。
  共置组和原共享池均为 32/32，两个静态组为 24/32。
- 12 pass 全部保留 512 条唯一 ID；源图 hash、尺寸、分辨率一致。完整导出回读
  校验通过；DINO 向量/范数差异为 0，质量评分最大绝对差异为
  `1.5497207641601562e-6`，低于固定的 `1e-4` 容差。
- 实机 OOM 次数为 0。没有因容差、缺失样本或失败运行而缩小验收范围。
- 结束后全部 8 卡为 0 MiB，无遗留 GPU 计算进程。

首次尝试因硬编码 256 MiB 解码批上限不足而被容量检查拒绝，完成 pass 为 0，
没有计入性能结果。之后修复实验预算推导，重新运行全部五组；失败日志完整保留。

## 产物与下一步

方案和使用方式：[elasticjuicer_compute_flow.md](elasticjuicer_compute_flow.md)。
真实算子和运行脚本位于 `tools/elasticjuicer/compute_flow_image_ops.py`、
`tools/elasticjuicer/benchmark_compute_flow.py`，不依赖其他实验目录代码。

本机可检查的证据：

- `/mnt/workspace/miaoxiang.zfr/test/compute-flow-review-20261027/audit.json`：六组交叉审计。
- 同目录 `audit.py`、`regression.log`：审计重放与 115 项回归日志。
- `/mnt/workspace/miaoxiang.zfr/test/compute-flow-validation-v2-20261027`：五组所有输出、时间线、源码 hash。
- `/mnt/workspace/miaoxiang.zfr/test/compute-flow-grouped-validation-20261027`：共置控制的全部原始产物。
- `/mnt/workspace/miaoxiang.zfr/test/compute-flow-validation-20261027` 及同名 `.log`：首次失败。

下一阶段优先在相同数据/同一物理分组上运行至少 3 个交错配对块，观察 CPU 解码、
传输、GPU 服务和导出压力，再决定调度策略是否值得开启；保留 legacy 与 grouped
基线。之后才扩展 20k/100k。物理 Actor 伸缩、模型卸载/重新放置、跨机资源交接和
持久化提交恢复需单独设计与验证。本次未恢复此前暂停的全量实验。
