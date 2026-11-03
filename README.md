# AI 玩超级马里奥

在 Super Mario Bros. World 1-1 上训练和评测 PPO 智能体。训练使用 84×84 的四帧灰度图和七个手柄动作。交付包提供可加载的模型、随机与训练后对局、实验记录、训练曲线和 3～5 分钟演示视频。

## 在 Windows 开始

准备 64 位 Python 3.11、Visual Studio 2022 C++ 桌面开发工作负载与 Windows SDK。录制 MP4 需要 PATH 中的 FFmpeg。Git 用于克隆和记录新的实验提交。

在项目根目录打开 PowerShell，依次运行三条命令。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\scripts\start_train.bat --config configs\delivery.yaml
.\scripts\start_eval.bat --model models\current\best_model.zip --episodes 10 --stochastic
```

第二条命令运行 16,384 步短训，完成后会更新当前模型。如果想保留交付包内的已训模型，可先评测该模型，或用 `--no-publish` 短训。配置 `configs/baseline.yaml` 保留百万步预算，其他 YAML 用于奖励对照和冒烟。评测 JSON 默认写入 `logs/<exp_id>/evaluation.json`，详情文件含逐局结果和内存峰值。

```powershell
.\scripts\start_demo.bat
```

交付包包含 `data/demo.mp4`。代码克隆中可以由已训练模型现场生成一局并播放。

## 命令行入口

```bash
python train/train_ppo.py --config configs/baseline.yaml
python eval/evaluate.py --model models/current/best_model.zip --episodes 10
```

默认模型评测取贪心动作。用 `--stochastic` 复现报告中的按概率采样策略。随机基线用 `--random`。录屏用 `--video data/play.mp4`。

- [环境与故障处理](docs/environment.md)
- [需求、架构与选型](docs/design.md)
- [训练续训和回放](docs/user-guide.md)
- [实验与验收报告](docs/results.md)
- [失败案例](docs/failure-cases.md)
- [独立测试清单](docs/test-plan.md)
- [演示分镜](docs/demo-script.md)

在 [Releases](https://github.com/ShiroRin-87/mario-ppo-world-1-1/releases) 下载完整项目 ZIP、演示成片、PPO 原始对局、随机对照和实测训练曲线。ZIP 含模型、训练日志与验收证据。
