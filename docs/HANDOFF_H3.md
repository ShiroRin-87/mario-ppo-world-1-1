# H3 评测与演示交接

- 交出岗位：IM-2
- 接收岗位：AS-2
- 接收日期：2026-11-03
- Git 标签：`h3-demo-ok`
- 评测提交：`1b6613d180a56b1d108fa0c0fe1268ec149cfa0d`
- 模型实验：`exp_20261103_death_02`

## 1 能跑的命令

从仓库根目录运行。交付模型置于 `models/current/best_model.zip`。

```powershell
python eval/evaluate.py --model models/current/best_model.zip --episodes 10 --seed 42 --stochastic --output data/verification.json
ffprobe -v error -show_entries format=duration data/demo.mp4
```

独立克隆实测最远距离 1959，平均奖励 613.0，十局通关率 0%。随机基线最远 898，平均奖励 460.1。随机及 PPO 对局 MP4 均可解码，成片时长 195 秒。

## 2 已知问题

- 达到随机最远距离的 2.18 倍，距离指标仍低于 3 倍目标
- 使用种子 42～51 选择模型。复核种子 4242～4251 的结果单列
- 演示循环播放每种策略最远的一局。评测结果另附十局原始记录

## 3 接手入口

组装 `data/demo.mp4`、`data/comparison/`、`logs/` 和 `models/current/best_model.zip`。README 的训练与评测入口使用现有脚本。

## 4 接收记录

- [x] 独立克隆复现模型指标
- [x] 核对两段实录及 195 秒成片可解码
- [x] 接收检查未修改上游源码
