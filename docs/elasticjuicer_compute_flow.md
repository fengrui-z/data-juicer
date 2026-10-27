# ElasticJuicer Compute Flow：双层调度开发方案

开发分支：`feat/elasticjuicer-compute-flow`。基于 Resident DAG v0.2.0
（`77885fb1333e6196deba143f0e0db9abc2351fdd`），作为该执行器的可选扩展。
这不是对 DJ 当前主库可以直接独立 cherry-pick 的补丁；提交主库 PR 时需要先纳入
Resident DAG 前置提交，或拆为前置执行器 PR 与本功能 PR。ZIP 待用户确认后打包。

## 可行性与实现边界

可行。现有执行器已经有固定大小的 Ray 常驻池、显式依赖与分支汇合、跨任务模型复用、
Actor 内独立 batch 控制，以及在途数量/估计字节预算。本次在其上加入全局执行组调度。
默认关闭，不改变已有 `ray`、`ray_partitioned` 入口，也不注册新的默认 CLI 后端。

启动时一次性创建 CPU 池与 GPU 池；每个 GPU Actor 占一张完整卡，加载该池涉及的
所有 DJ Mapper。全局控制器根据可执行队列、在途工作量和实测每行耗时估算各执行组
所需算力，分配目标并发。Actor 完成当前组后，可以领取另一个组的任务。
模型和各算子的 batch 学习状态留在原 Actor 中，不随任务调度搬运。

```text
                   全局 ComputeFlowController
              就绪积压 / 等待时间 / 实测服务速率
                          ↓ 组并发目标
             已创建的 Ray GPU Actor 池（例如 8 个）
               Actor 0 … Actor 7：空闲时领取执行组
                          ↓
             每 Actor、每算子的 AdaptiveBatchController
               成功探测增长 / OOM 缩批 / 安全重试
                          ↓
                 DJ Mapper → DAG 汇合 → 输出
```

**这里的流动是固定池内的任务执行算力流动。** 不销毁/重建 Actor，不动态改变
Ray 声明的 CPU/GPU 数量，不自动卸载或迁移模型，也不把减少某组任务误报为释放显存。
多个模型必须能共驻于单卡；单个执行组的峰值激活加模型常驻量也必须可承受。
模型集合过大时，应按显存分池，跨池模型副本交接留到后续版本。

## 两层如何配合

| 层 | 控制对象 | 反馈与策略 | 明确边界 |
|---|---|---|---|
| 全局 | 空闲 Actor 的执行组租约、组并发目标 | 待处理行数 × EWMA 每行耗时；在途任务纳入需求；周期更新、整数迟滞 | 不改算子 batch，不抢占在途任务 |
| 局部 | 每 Actor、每算子的 microbatch | 复用已有自适应控制器与 OOM 安全切片；可从小 batch 起步 | 不增减 Actor，不覆盖其他 Actor 的成功/OOM 历史 |

目标是软并发目标：旧任务先完成；目标已满但有空闲卡时可以借出执行能力，避免空转。
等待时间超过阈值的组优先获得下一次空闲机会，防止少卡多组时饿死。组清空后的尾部
资源能自动流向仍有工作的一组。首个非空 Actor/组样本不参与速率估计，避免模型加载/
CUDA 首次调用成为稳定吞吐先验；空批不污染画像。全局反馈使用 Worker 单调时钟的
完整组耗时，包含父数据还原与局部重试，不含 driver 等待或下游消费耗时。

现有 DAG 的精确 ID/版本汇合、subset 契约、delta ObjectRef 传输、在途预算、
暂停排空、超时与失败关闭自有池机制继续生效。物理显存预算、持久化 checkpoint、
提交 ACK、Actor 故障自动重建、多机调度、跨池自动放置仍不在本版承诺范围。

## 接入方式

```python
from data_juicer.core.elasticjuicer.compute_flow import ComputeFlowConfig
from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor

# 调用者先 ray.init；plan 仍使用 DJ operator recipe、DAGNode、ExecutionGroup、WorkerPool。
# pools=() 默认选择所有 GPU 池；可以显式填入 CPU 池名。
flow = ComputeFlowConfig(interval_seconds=0.5, starvation_seconds=5.0)
with ResidentDAGExecutor(plan, adaptive=True, compute_flow=flow,
                         artifact_dir="new-run-directory") as executor:
    for batch in executor.iter_batches(input_batches):
        export(batch.columns())
```

`DAGNode.batch_size` 是 microbatch 上限；可设置 `initial_batch_size` 作为自适应模式
起点，默认等于上限。固定模式始终使用上限。新增参数不会让全局控制器获得 batch
控制权。现有算子仍须显式满足 `elastic_juicer_dag_safe` 契约。

`ComputeFlowConfig(mode="static")` 按固定 Actor slot 分组；默认均分，可用
`initial_slots=(("quality", 2), ("embedding", 6))` 指定。每个选定池必须完整配置，
数量之和等于该池 Actor 数；静态模式每组至少一个。动态模式可以组多于 Actor，
靠任务边界轮转保证推进。`compute_flow=None` 保留原来的下游优先、FIFO 空闲复用。

## 实验：真实 CPU/GPU 多算子与负载变化

```text
CPU Decode：读取、SHA256 校验、RGB 解码
       ├── GPU QualiCLIP：224，FP32，质量评分 ─┐
       └── GPU DINOv2-large：FP16，特征提取 ──┤
                            精确 ID + 图像 hash 汇合
                                      ↓
CPU Pack：检查评分/1024 维向量、计算范数 → 流式导出 NPZ
```

两条 GPU 分支独立，可以同时在不同 Actor 上执行。CPU 与 GPU 对不同 batch
形成流水；所有模型真实推理，没有 sleep 或伪造预测。`phase-shift` 使用前半段
448、后半段 896 的 DINO 输入；这是实际图像特征计算量变化，所有对照采用相同规则。
这项实验不用于比较这两种分辨率的模型质量。

| 对照 | 全局分配 | 局部 batch | 用途 |
|---|---|---|---|
| legacy-fixed | 原 v0.2 共享池调度 | 固定 | 检验新策略相对现有模块的增量 |
| grouped-fixed | 一个常驻 GPU 执行组内并行跑两个算子 | 固定 | 检验已有算子共置与组内并行的收益 |
| static-fixed | 固定 slot 归属，8 卡默认 4/4 | 固定 | 固定资源基线 |
| flow-fixed | 动态组租约 | 固定 | 隔离全局层效果 |
| static-adaptive | 固定 slot 归属 | 自适应 | 隔离局部层效果 |
| flow-adaptive | 动态组租约 | 自适应 | 验证双层协作 |

六组使用相同 Actor 数、CPU/GPU 声明、预加载模型、逻辑 DAG、输出语义、microbatch
上限和在途预算。`grouped-fixed` 单独改变物理执行分组，使用 Resident DAG 的
`ExecutionGroup(parallel=True)`，不是 DJ 主库的 `FusedSequentialBatchOp`，也不
代表自动融合搜索。静态组也预加载两模型，防止把模型启动/驻留差异误归因于调度。
另报这种共驻方案相对专用单模型 Actor 的显存代价；本实验不代表二者的全部成本对比。

主指标分别报告“建池至首次完整导出”和“同池后续 warm pass”；Ray 启动不计入上述
两个区间，后验正确性比对也不计入。所有输出完整导出，按 ID 校验数值（绝对容差
`1e-4`）、尺寸、图像 hash、输入分辨率；记录每张实际 GPU、换组次数、实例复用、
分支重叠、实际 batch/OOM 和目标分配时间线。不能仅凭 GPU utilization 宣称提速。

默认 512 张、每组 2 pass、1 个实验块用于机制验收。正式性能结论至少 3 个配对块，
每块轮换模式顺序；之后再扩至 20k。100k 必须明确为 20k 原图 × 5 个不同记录 ID，
不是 100k 独立图片；不得把该吞吐规模试验用作模型质量数据扩容。

运行脚本自包含于本分支，不依赖其他实验目录代码；需要本地数据、可信 QualiCLIP
定义及权重、DINOv2-large 权重和 DJ/Ray/PyTorch/torchvision/transformers 环境：

```bash
python tools/elasticjuicer/benchmark_compute_flow.py \
  --input /path/to/images.jsonl --qualiclip /path/to/QualiCLIP \
  --dinov2 /path/to/dino-v2 --output /path/to/new-run \
  --gpus 8 --rows 512 --passes 2 --repeats 1
```

JSONL 每条包含唯一 `id`、`image_path`、`image_sha256`。脚本先验证源文件与可用
GPU，并读取实际图片尺寸，按保守的 RGB 解码占用推导每批上限。总对象预算默认
64 GiB（可用 `--object-budget-gib` 修改），在途上限默认 16（`--inflight`）；预算
不足时执行器降低有效在途数量，连一批也无法容纳则在建池前报错。仅创建/关闭自己的
本地 Ray 集群，不停止其他任务。原暂停的 20k/100k 实验
不因此恢复。代码、环境、输入 hash 与计划记录在 `provenance.json`；`results.json`
保存耗时/校验结果，每个 cell 的 `compute_flow.jsonl` 保存计划与实际租约证据。

## 潜在收益与需要实测的风险

在算子耗时不均衡、工作量变化或收尾时，可让原先绑定轻算子的空闲卡接手重算子。
局部缩批保证可执行，全局调度则基于实际服务速率补充该阶段的执行机会。已有融合组
仍可整体调度，保留融合收益。

若原有共享池调度已经高效，新增决策层可能无收益甚至略增开销；CPU 解码、数据传输
或导出受限时也不能靠重分 GPU 解决。所有模型共驻有显存代价，短任务的调度开销和
测量噪声更明显。上层软目标有冷却与等待保护，但不是硬实时服务保证。最终以完整
作业耗时和有效输出吞吐判断；实现/回归通过不等于已证明 20k/100k 加速。
