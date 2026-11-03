# 训练与回放操作

首次使用先按 README 安装。新训练的模型保存在 `models/current/best_model.zip`，每轮的 `logs/<exp_id>` 包含配置、实验登记、模型存档、训练日志和 TensorBoard 数据。

## 续训

用要恢复的 checkpoint 和配置启动。配置中的步数是此次新增步数，实验登记会保留起始步数。

```powershell
scripts\start_train.bat --config configs\baseline.yaml --resume logs\<exp_id>\checkpoints\ppo_100000.zip
```

`--no-publish` 将实验模型留在该轮日志目录。快速检查时使用此参数，可继续用当前交付模型进行评测。

## 对比与录像

```powershell
scripts\start_eval.bat --random --episodes 10 --seed 42 --output logs\comparison\random.json --video data\random.mp4
scripts\start_eval.bat --model models\current\best_model.zip --episodes 10 --seed 42 --stochastic --output logs\comparison\ppo.json --video data\ppo.mp4
scripts\start_demo.bat
```

对比使用相同的关卡、局数、种子和基础环境奖励。训练奖励改动通过配置设置。评测 FPS 衡量每秒策略决策步，录像保留每四个模拟器帧中的一帧，以正常游戏速度播放。录像选取该组最远的一局复跑生成。
