# 最终交付记录

AS-2 交 PM。项目包从已提交的源码生成，包含 CPU 模型、实验记录、3 分 15 秒成片、原始对局和报告。源码提交见包内 `SOURCE_COMMIT`，模型身份见 `data/manifest.json`。

## 1 能跑的命令

在解包目录按 README 执行安装、训练和评测。若只想看已训练模型，完成安装后运行评测和演示。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\scripts\start_eval.bat --model models\current\best_model.zip --episodes 10 --stochastic
.\scripts\start_demo.bat
```

验收在新解包目录复用已验证的隔离 CPU 环境。安装脚本的依赖检查及 100 步冒烟通过。启动脚本短训 512 步，加载模型归档内的存档续训至 1024 步。十局复评仍得到最远 1959、平均奖励 613.0。成片完整解码成功。

## 2 已知问题

- 交付模型最远距离为随机的 2.18 倍，低于 3 倍目标
- 百万步配置随包交付，完整训练耗时尚无实测
- Windows 复跑已验证，WSL2 方案见环境文档

## 3 材料位置

| 项目 | 位置 |
|---|---|
| 可运行系统与源码 | 项目根目录、`scripts/`、`models/current/`、GitHub 仓库 |
| 训练记录与曲线 | `logs/`、`data/comparison/*_curve.png` |
| 基线与对照 | `docs/results.md`、`data/comparison/` |
| 测试与失败案例 | `docs/test-report.md`、`docs/failure-cases.md`、`data/test-evidence/` |
| 成片、分镜及实录 | `data/demo.mp4`、`docs/demo-script.md`、`data/comparison/*.mp4` |
| 工作记录与输出文档 | `docs/work-log.md`、`docs/design.md`、`docs/environment.md`、`docs/user-guide.md` |
