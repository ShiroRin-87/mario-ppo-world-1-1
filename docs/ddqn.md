# Double DQN 检查点评测

评测入口支持从 MarioPlayingAgent 压缩包导入 `runs/checkpoints/mario_ddqn.pt`。它使用检查点的在线网络做贪心推理，同时兼容只保存模型参数的同结构权重。

在项目已有的训练环境中补装 OpenCV，并把权重放到本地 `models/current/`。

```powershell
python -m pip install "opencv-python-headless==4.11.0.86"
python eval/evaluate.py --ddqn models/current/mario_ddqn.pt --episodes 10 --seed 42 --output logs/ddqn/evaluation.json --video logs/ddqn/replay.mp4
```

评测复用项目的 World 1-1 模拟器、动作表、4 帧跳帧和回合上限。DDQN 使用原权重所需的灰度系数和 OpenCV 面积插值，输入为四帧 84×84 图像。附件的训练环境使用较新的模拟器，成绩以当前项目环境的复跑为准。

以下命令用相同种子范围和贪心模式评测 PPO。

```powershell
python eval/evaluate.py --model models/current/best_model.zip --episodes 10 --seed 42 --output logs/ppo/evaluation.json --video logs/ppo/replay.mp4
```

两种模型均输出汇总、逐局 JSON 和 MP4。权重从本地读取，训练入口仍使用各自的工程。

## 附件权重复跑

种子 42–51，双方均取贪心动作。各模型在这十次运行中的成绩一致。

| 模型 | 通关 | 最远横向坐标 | 平均奖励 |
| --- | --- | --- | --- |
| Double DQN（附件） | 0/10 | 1242 | 1122 |
| PPO（项目交付） | 0/10 | 314 | 250 |

这组结果比较两份指定权重在当前环境中的表现。
