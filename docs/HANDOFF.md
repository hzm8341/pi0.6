# 交接文档（2026-11-14，停电前）

新会话接手前请先读：`docs/DECISIONS.md`（已确认的决定）→ `docs/RECAP_PLAN.md`（计划与进度）→ 本文件。

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
