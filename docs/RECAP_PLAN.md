# RECAP 开发计划（基于 π0.5）

> 制定日期：2026-11-14 ｜ 依据：π\*0.6 论文（arXiv 2511.14759）、RLinf RECAP 文档、RECAP-on-YAM 复现、本仓库现状评估
> 决策记录见 `docs/DECISIONS.md`；未完成项的旧清单见 `TODO_RECAP.md`（本文件取代其优先级）。

## 0. 目标与边界

**目标**：在 π0.5（PaliGemma 2B + 300M action expert）上实现完整、可验证的离线 RECAP 闭环，并在仿真（LIBERO）和真机（试管/试管架任务）上给出"π0.5 SFT vs RECAP"对照数字。

**不做**：Gemma3 4B 主干升级、完整 KI + FAST 联合目标（无公开权重，投入产出比低）。

**特色方向**（区别于 RLinf / pistar / pi-openpi）：

| 编号 | 特色 | 说明 |
|---|---|---|
| F1 | 共享主干 value head | 在 π0.5 prefix 输出上挂 201-bin 分布式 value head，单模型同时出动作与价值，单卡可训 |
| F2 | Value 作为运行时监控 | 部署时输出 V(o_t)，下降趋势触发减速/停机/请求人工接管，并与 HITL 采集联动 |
| F3 | RECAP × MEM | value / policy 使用视频与状态历史，改善进度估计（停顿 vs 推进） |
| F4 | 分阶段 value / reward | 试管任务按 接近→抓取→提起→插入 分阶段，结合力/接触信号 |
| F5 | 低成本 CFG | advantage token 因果化 → 条件/无条件共享 KV cache，β 消融 |
| F6 | 标签与数据同存 | 标签按 LeRobot 全局帧索引存储，可不重跑 GPU 改阈值重打标 |

## 1. 现状问题（评估结论，2026-11-14）

| # | 问题 | 位置 | 严重度 |
|---|---|---|---|
| P1 | 推理时不注入 positive 条件，部署策略≈无条件 SFT | `transforms.py` `TokenizeReCAPAdvantage` | 致命 |
| P2 | 代理 value 下成功轨迹 advantage 恒为 0，退化为"按成败过滤" | `training/recap_value_proxy.py` | 致命 |
| P3 | `Observation` 中 advantage token 与 prompt 共用 jaxtyping 维度名 `l`，长度不同时类型检查报错（3 个测试失败） | `models/model.py` | 高 |
| P4 | RECAP loss 两次完整前向，训练开销约 2× | `models/pi0.py` `compute_recap_loss` | 中 |
| P5 | advantage token 在双向 prefix 中，图像/prompt 能看到它；无法共享 KV 做 CFG | `models/pi0.py` `embed_prefix` | 中 |
| P6 | 无 CFG 推理 | `models/pi0.py` `sample_actions` | 中 |
| P7 | 无可学习 value 模型、无 value 训练/推理 | `models/value_function.py` | 高 |
| P8 | npz 旁路标签依赖帧顺序对齐 | `training/data_loader.py` / `recap_offline.py` | 中 |
| P9 | 无 SFT vs RECAP 对照评估、无 rollout/HITL、编排脚本只打日志 | `scripts/recap_train.py` | 高 |
| P10 | 本机 `uv sync` 因 `av` 仅有 sdist 需编译 FFmpeg 而失败 | `uv.lock` | 低（用 `uv sync --no-install-package av` 绕过跑单测） |

## 2. 分阶段计划

### Phase 1：修正条件化训练与推理（P1 P3 P4 P5 P6）

- [x] 1.1 修 `Observation` 维度名：advantage token 改用独立维度 `al`
- [x] 1.2 advantage token 因果化：在 prefix 中 `ar_mask=[True, False, ...]`，图像/prompt 看不到它，动作 token 能看到
- [x] 1.3 推理默认条件：`TokenizeReCAPAdvantage(default_indicator=True)`，样本缺少 `advantage_indicator` 时按 positive 处理（与论文"SFT 示范 I=True"一致）
- [x] 1.4 单次前向 loss：`ReCAPConfig.loss_mode = "single"`（默认，随机丢弃条件）/ `"dual"`（保留论文式两项）
- [x] 1.5 CFG 推理：`ReCAPConfig.cfg_beta`（默认 1.0 = 仅 positive 条件）；β>1 时 `v = v_u + β (v_c − v_u)`，前缀 KV 只算一次，条件/无条件分支只差 prefix 掩码
- [x] 1.6 单元测试：类型、因果掩码、默认条件、单/双 loss、β=1 与条件采样一致、β>1 可运行

**验收**：RECAP 相关测试全部通过；`debug_pi05_recap` 能训练几步并推理。

**附带修复（开发中发现的既有问题）**：
- `RepackTransform` / 平台 `*Inputs` 会丢弃 sidecar 字段，advantage 标签从未进入模型 → 新增 `PreserveFields` 包裹 repack + data transforms
- `config.py` 中 `default_factory=GroupFactory` 实例化 Protocol，在 Python 3.11.16 下导入即报错（所有训练脚本不可用）→ 改为 `EmptyGroupFactory`
- `Observation.from_dict` 对 `image_history=None` 调用 `.items()`，FakeDataset 训练报错 → 改为判空
- advantage 文本改为无 BOS 的原始编码，并校验 positive/negative 等长

**已知问题**：
- MEM 相关 12 个测试在 master 上即失败（`test_memory_manager` / `test_video_encoder` / `test_policy_mem`），与 RECAP 无关，未处理
- value 大量相同时，分位数阈值的 `>=` 会使 positive 比例远超设定（`assign_advantage_labels`），后续可加随机打破平局或严格大于
- 本机 `av==14.4.0` 无 wheel；测试环境用 `uv sync --no-install-package av` + `uv pip install --only-binary av "av>=14.4,<17"`（不改 uv.lock）

### Phase 2：可学习 value 模型（P2 P7，特色 F1）

- [x] 2.1 `Pi0` 增加可选 `value_head`（`ValueHead` MLP）（`ReCAPConfig.value_head=True`）：读取 advantage token 之前的 prefix 表征（有效 token 均值池化）→ MLP → 201 bins
- [x] 2.2 value 目标：按论文 Eq.5 的 reward（每步 −1，失败终止 −C_fail），按任务最大长度归一化到 (−1, 0)，离散成 bins 后用交叉熵训练
- [x] 2.3 训练模式：`value_only`（冻结或低 lr 主干，只训 value，用于标注）/ `joint`（policy loss + λ·value loss，对主干 stop-grad 可选）
- [x] 2.4 数据：数据集额外提供 `value_target` 字段（由标注脚本基于 episode 成败与长度生成）
- [x] 2.5 `scripts/label_recap_with_value.py`：批量推理 V(o_t) → 计算 N 步 advantage → 分位数阈值（人工干预强制 positive）→ 写标签 + `value_label_summary.json`
- [ ] 2.6 指标：留出 episode 上 Spearman(V, return)、成功/失败 episode 的 V 曲线分离度（脚本已输出全量 Spearman / MAE；留出集评估与曲线图待做）
- [ ] 2.7 在真实数据上跑通：`pi05_recap_value` 训练 → 打标签 → `pi05_recap` 训练（需 GPU 服务器与 pi05_base 权重）

**验收**：在 G1 或 LIBERO 数据上 Spearman > 0.8；成功轨迹中快/慢片段能得到不同 advantage。

### Phase 3：标签存储与数据通路（P8，特色 F6）

- [ ] 3.1 标签按 `(episode_index, frame_index)` 存储（parquet），加载时按键 join，不依赖顺序
- [ ] 3.2 `relabel` 工具：已存连续 advantage，改阈值/N 时不需重跑 GPU
- [ ] 3.3 LeRobot 数据集直接作为 RECAP 输入（不再要求 episode JSON）

### Phase 4：对照评估（P9 前半）

- [ ] 4.1 LIBERO-10 复现 RLinf 设置：few-shot π0.5 SFT → rollout 采集 → value → RECAP；目标复现 48.8%→66.5% 同量级提升
- [ ] 4.2 评估脚本：成功率、平均完成时间、失败类型、标签分布、β 消融（1 / 1.5 / 2 / 3）
- [ ] 4.3 `scripts/recap_train.py` 串联真实子进程（标注→value→训练→评估→报告）

### Phase 5：真机 HITL（P9 后半，特色 F2 F4）

- [ ] 5.1 `rollout/recap_env.py`、`episode_recorder.py`、`intervention.py`、`safety.py`
- [ ] 5.2 Value 运行时监控：滑动窗口 V 下降阈值 → 减速 / 停机 / 请求接管
- [ ] 5.3 分阶段成功标签与力/接触信号接入 reward
- [ ] 5.4 每轮 ≥100 条 rollout；每轮从预训练（SFT）权重重训 value 与 policy（论文做法）

### Phase 6：RECAP × MEM（特色 F3）

- [ ] 6.1 value head 读取视频记忆融合后的图像 token 与状态历史
- [ ] 6.2 对照：有/无记忆的 value 相关系数与 RECAP 成功率

## 3. 进度日志

| 日期 | 内容 |
|---|---|
| 2026-11-14 | 完成评估，制定本计划；开始 Phase 1 |
| 2026-11-14 | Phase 1 完成（1.1–1.6）；Phase 2 代码完成 2.1–2.5（共享主干 value head、`pi05_recap_value` 配置、`label_recap_with_value.py`），待真实数据验证 |
