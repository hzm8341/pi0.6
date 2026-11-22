# 交接文档（2026-11-14，停电前）

新会话接手前请先读：`docs/DECISIONS.md`（已确认的决定）→ `docs/RECAP_PLAN.md`（计划与进度）→ 本文件。

## 0. 周一（11-23）回来先看这里：休假期间结果汇总

LIBERO-10 Task 0。前几行是 50 次评估（标准误约 7%），最后一列是补到 200 次（episode 0–199，标准误约 3.5%）：

| 方法 | 50 次 | 200 次 |
|---|---|---|
| SFT（10 条示范，2.5k 步） | 52% | 50.5% |
| 过滤 BC（示范 + 273 条成功 rollout，2.5k 步） | 62% | 56.0% |
| RECAP r1（价值标签，2.5k 步，β=1 / 1.5 / 2） | 54% / 38% / 48% | — |
| RECAP r1 续训到 5k 步（β=1 / 1.5 / 2） | 32% / 22% / 24% | — |
| RECAP r1 负条件（2.5k / 5k） | 44% / 26% | — |
| RECAP r2（成功 episode 门控标签，2.5k，β=1 / 1.5） | 54% / 58% | — / **57.0%** |
| RECAP r3（奖励 v2 + value v2 标签，2.5k，β=1 / 1.5） | — | 46.0% / 53.5% |

- r1 的 positive 帧里 41% 来自失败 rollout（奖励截断在 [-1, 0]，失败惩罚被截没）。r2 把失败轨迹全标 negative 后 CFG 起正作用。
- 奖励 v2（用户 11-21 决定）：失败惩罚不再截断。代理标签能完全分开成功/失败（失败 rollout 0% positive），但用 value v2 打的学习版标签仍有 36.9% 的 positive 来自失败 rollout，r3 没有改善。问题在 value 模型的 N 步优势估计，不在奖励截断。
- 200 次结论：r2 ≈ 过滤 BC（57% / 56%）> SFT（50.5%），差约 6 个百分点，接近 2 倍标准误。
- 全量微调（用户决定本机方案 B）：11-22 03:09–13:09 训到 2.5k 步（13.9 秒/步，loss 降到 0.004），200 次评估 **44.0%**，低于 LoRA SFT 的 50.5%，可能是在 10 条示范上过拟合。续训到 5k 没做：第一次恢复 OOM（已修），之后 Abaqus 作业跑到 11-23 01:40，剩余时间不够在 10:00 前跑完，已取消，GPU 空闲。要续训：`RESUME=1 STOP_STEP=5000 scripts/libero/run_full_ft_sft.sh`（约 10.5 小时，会先等 Abaqus、要求可用内存 ≥70GB）。
- 可做的下一步：value 的留出集评估；改 value 的优势估计（例如用回合结果门控或更长的 N）。

## 1. 当前状态一句话

Phase 1（推理条件 / CFG / 单次前向 loss）和 Phase 2（π0.5 共享主干 value head + 学习版打标签）代码已完成并推送到分支 `feat/recap-phase1`（commit `48f7f19`，未开 PR）。正在开始 Phase 4：LIBERO-10 Task 0 上做 π0.5 SFT vs RECAP 对照，环境准备刚起步。

## 2. 已确认的决定（详见 DECISIONS.md）

- 基于 π0.5 修改，不升级 Gemma3 4B。
- 优先级：推理条件 + CFG → 学习版 value → LIBERO 对照 → 真机。
- LIBERO 用原生 MuJoCo/robosuite 版（与 RLinf 可比）；Isaac Sim 5.1 留给以后试管任务自建场景（此条为 Claude 建议，用户未明确反对）。
- 第一轮规模：Task 0、500 条 rollout、SFT/value/RECAP 各训练 10k 步，单卡 5090 + LoRA。

## 3. 未提交的改动

- `docs/DECISIONS.md`：新增 3 条（提交推送、LIBERO 仿真建议、实验规模）。
- `docs/RECAP_PLAN.md`：新增"Phase 4 实验设计"小节（步骤 A–G）。
- 本文件。

> 如果停电前来不及提交，这三个文件就留在工作区，开机后 `git status` 可见。

## 4. 推送方式

`origin` 是 HTTPS，本机没有凭据，`git push` 会失败。用 SSH 地址推送（已验证可用）：

```bash
git push git@github.com:hzm8341/pi0.6.git feat/recap-phase1
```

## 5. 本机环境（重要坑）

- `uv sync` 会失败：`av==14.4.0` 在本机无 wheel、需要编译 FFmpeg。已用的绕过方式（不改 `uv.lock`）：
  ```bash
  uv sync --no-install-package av
  uv pip install --python .venv/bin/python --only-binary av "av>=14.4,<17"   # 装到了 av 16.1.0
  ```
  注意 `uv sync` 曾自动改写 `uv.lock`，已 `git checkout uv.lock` 还原；以后跑完 sync 记得检查。
- 跑测试：
  ```bash
  .venv/bin/python -m pytest -q src/openpi/training/test_recap_*.py src/openpi/models/test_recap_*.py \
      src/openpi/training/data_loader_test.py -k "not real_dataset"
  ```
  `model_test.py::test_model_restore` 会从 GCS 下大权重、看起来像卡死，要排除（`-k "not test_model_restore"`）。
- 已知与 RECAP 无关、master 上就失败的 12 个 MEM 测试：`test_memory_manager` / `test_video_encoder` / `test_policy_mem`。
- 最近一次完整回归：104 passed，13 failed（12 个 MEM + 1 个 CFG 测试的 GPU TF32 精度问题，已加 `jax.default_matmul_precision("highest")` 修复，单独复跑通过）。
- GPU：RTX 5090 32GB → 只能 LoRA 训练 π0.5。
- `pkill -f "<pattern>"` 会把自己的 shell 也杀掉（命令行里含同样字符串），别这么用。

## 6. Phase 4 进行到哪了

已完成：
- LIBERO 源码已克隆到 `third_party/libero`（`.gitmodules` 有登记但 git 索引里没有 gitlink，所以 `git submodule update` 无效；已在本地 `.git/info/exclude` 忽略 `third_party/`）。
- 确认 libero_10 Task 0 = "put both the alphabet soup and the tomato sauce in the basket"，在 `physical-intelligence/libero` 里 `task_index=5`，共 33 条示范。
- 数据集元数据已下到 `/media/hzm/data24t/datasets/libero_src/meta/`。

2026-11-16 更新：
- 33 条 Task-0 示范已下载完（`/media/hzm/data24t/datasets/libero_src`，`task0_episodes.json`）。
- `pi05_base` 已下完（12GB）。gcsfs 不走代理会卡死，改用 `requests` + 代理分段续传才下完。
- Phase 4 代码已写完并在小数据上跑通，完整命令见 `docs/recap_libero.md`（步骤 A–G）。
- Docker daemon 已配代理（2026-11-17）；`libero` 镜像已建成并验证能创建 Task 0 环境、渲染图像。Dockerfile 修了两处：去掉 LIBERO 自带的 robosuite==1.4.0 约束（与 openpi 的 1.4.1 冲突），sdist 构建限制 setuptools<70（Python 3.8）。
- 步骤 A 完成（demos33 / demos10 数据集、`assets/libero_task0/norm_stats`）；步骤 B（SFT K=10）2026-11-17 11:00 开始，约 3.4 s/步，10k 步约 9.5 小时，日志 `logs/train_sft_k10.log`。
- 2026-11-17 晚：SFT@2.5k = 52%；500 rollout 54.6% 成功；value@2.5k Spearman 0.907；学习版标签在 `outputs/libero_task0/labels_value_s2500/lerobot_fields.npz`。暂停中（21:00 后 GPU 给其他任务）。恢复时直接跑步骤 F：
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_recap_libero_lora --data.recap-fields-path outputs/libero_task0/labels_value_s2500/lerobot_fields.npz --exp-name r1 --overwrite --no-wandb-enabled`

给 Docker daemon 配代理（需 sudo）：
```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
printf '[Service]\nEnvironment="HTTP_PROXY=http://127.0.0.1:7890" "HTTPS_PROXY=http://127.0.0.1:7890" "NO_PROXY=localhost,127.0.0.1"\n' | sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf
sudo systemctl daemon-reload && sudo systemctl restart docker
```

LIBERO 客户端推荐用 Docker（`examples/libero/compose.yml`，Python 3.8 + cu113 本地装很麻烦）；策略服务可直接用本仓库 `.venv` 跑 `scripts/serve_policy.py`。Docker 29.8 + NVIDIA CDI 已就绪。

## 7. 已知问题（来自 RECAP_PLAN.md）

- value 大量相同时，`assign_advantage_labels` 用 `>=` 阈值会让 positive 比例远超设定（假数据上 30% → 87.5%）。
- 还没有在任何真实数据上训练过，RECAP 是否有效尚无证据。

## 8. 用户休假期间的自主执行计划（2026-11-18 晚 → 2026-11-23）

用户授权期间不咨询、自行使用 GPU（见 DECISIONS.md）。GPU 由 T-Rex 后训练占用至约 11-19 中午，不要停它；等它结束且 `nvidia-smi` 无进程再开始。按顺序：

1. `scripts/libero/full_ft_trial.sh`：单卡全量试跑（约 30 分钟 + 权重转换）。结果（显存峰值、每步耗时、loss、梯度累积是否正常）写进 DECISIONS.md；按 `docs/pi05全量训练_单卡5090方案.md` 第 3 节判断方案 A / B / 租 A100，并估算正式训练总时长。
2. `scripts/libero/run_step_f_g.sh`：RECAP 训练到 2.5k 步 + β=1/1.5/2 评估（约 3 小时），与 SFT 52% 对比。
3. 过滤 BC 消融：demos10 + 成功 rollout → `local/libero_task0_filtered`，`pi05_libero_lora_fewshot` 训到 2.5k 步并评估（约 2.6 小时）。
4. 视结果：RECAP 不如预期时续训到 5k 步再评估；有余力再做留出集 value 评估（Phase 2.6）、第二轮 RECAP 迭代（用 π_RECAP 采新 rollout）。
5. 每完成一步更新 RECAP_PLAN 进度日志并提交推送。

**2026-11-18 21:11 状态：** 内存耗尽（可用约 1 GB，swap 63/63 GB 用满；主要是 T-Rex PID 3700589 约 69 GB 加它的 pt_data_worker），Claude Code 把本会话所有后台任务都停了：GPU 等待链（`wait_gpu_free.sh` → `full_ft_trial.sh` → `run_step_f_g.sh`）和截止守卫（`deadline_guard.sh`）。这些任务当时都还在等待，GPU 上没有跑任何本项目作业。按 Claude Code 的规则，被这样停掉的任务不能自行重启，要等用户要求，所以自主计划在这里暂停。恢复方法：在 Claude Code 里让我重启，或手动运行
`scripts/libero/deadline_guard.sh > logs/deadline_guard.log 2>&1 &` 和
`scripts/libero/wait_gpu_free.sh && scripts/libero/full_ft_trial.sh > logs/full_ft_trial_driver.log 2>&1; scripts/libero/run_step_f_g.sh > logs/run_step_f_g.log 2>&1`
