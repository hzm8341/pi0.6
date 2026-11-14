# 决策记录

格式：日期 ｜ 决定 ｜ 原因 ｜ 决定人。推翻旧决定时不删除，在条目后注明改动。

- 2026-11-14 ｜ RECAP 继续开发，以 π0.5（PaliGemma 2B + 300M action expert）为基座做修改，不追求论文级 π0.6（Gemma3 4B / 860M expert） ｜ 用户倾向；π0.6 权重未公开，算力成本高 ｜ 用户
- 2026-11-14 ｜ 下一阶段优先补齐 ① 推理时注入 positive 条件 + CFG ② 可学习的 value model 替换进度代理 ③ 与 π0.5 SFT 的对照评估（先 LIBERO 仿真，再真机） ｜ 现有代理 value 使成功轨迹 advantage 全为 0，推理端未注入条件 ｜ Claude 建议，用户 2026-11-14 确认（"好的"），计划见 `docs/RECAP_PLAN.md`
- 2026-11-14 ｜ Phase 1/2 改动提交并推送到 `feat/recap-phase1`；下一步做 Phase 4 LIBERO 对照（SFT vs RECAP） ｜ 先拿到可与 RLinf 对比的数字 ｜ 用户
- 2026-11-14 ｜ 建议（未确认）：LIBERO 对照使用原生 MuJoCo/robosuite 版 LIBERO（与 RLinf 设置一致），Isaac Sim 5.1 留给后续试管任务的自建场景 ｜ LIBERO 本身基于 MuJoCo，换仿真器无法与 RLinf 数字对比 ｜ Claude 建议
