# H2 训练交接

- 交出岗位：IM-1
- 接收岗位：IM-2
- 接收日期：2026-11-03
- Git 标签：`h2-model-ok`
- 上游提交：`2640fb9dbbbb2c68513c8a5bc663a2e4b7e94749`
- 交接模型：`exp_20261103_delivery_01/best_model.zip`

## 1 能跑的命令

在项目根目录执行。使用环境依赖锁和训练依赖锁安装 Python 3.11 的 CPU 环境。交接时权重在日志目录，当前代码已将权重统一放到 `models/archive/<exp_id>/`。将该实验的模型放入 `models/current/best_model.zip`。

```powershell
python -m unittest discover -s tests -v
python train/train_ppo.py --config configs/smoke.yaml --exp-id exp_20261103_cpu_smoke_01 --no-publish
python train/train_ppo.py --config configs/smoke.yaml --exp-id exp_20261103_cpu_resume_01 --resume models/archive/exp_20261103_cpu_smoke_01/checkpoints/ppo_512.zip --no-publish
python train/train_ppo.py --config configs/delivery.yaml --exp-id exp_20261103_delivery_01 --no-publish
```

实测五项环境测试通过。CPU 冒烟 512 步，加载 512 步存档后续训到 1024 步。交接实验运行 32,768 步，墙钟 68.022 秒。接收方在单独克隆中用 SB3 加载模型，复位 World 1-1，得到 `PASS loaded action=4 reward=0.0 x_pos=40`。

## 2 已知问题

- 交接模型用于确认可加载和评测。策略表现由 IM-2 和 QM-1 另测
- 当前验证范围含 Windows CPU 训练与 RGB 帧。百万步配置未在本交接跑完
- 日志与模型保存在交付包，按实验编号核对配置与记录

## 3 接手入口

评测脚本从 `eval/` 开始，加载 SB3 PPO 检查点，环境从 `envs/mario.py` 构造。对照同一套奖励和种子运行随机与训练后策略。

## 4 接收记录

- [x] 独立克隆加载模型并执行动作
- [x] 接收检查未修改训练与环境文件
