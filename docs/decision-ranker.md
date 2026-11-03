# 小模型动作排序实验

这个实验给 World 1-1 的采样 PPO 加一个本地动作排序器。模型为 17 → 32 → 1 的 MLP，共 609 个参数。训练标签来自实际模拟器预演，推理可离线运行。

## 训练与对照

每 12 个决策步尝试六种候选动作序列。独立模拟器从与主对局一致的状态预演每个候选。特征包含当前状态、动作序列编号与前 6 步的存活、位移和奖励。监督标签根据 16 步后的存活、距离和旗标计算。采集轨迹混用长预演排序、短预演规则和 PPO。

三组对照采用同一 PPO 权重、评测关卡、种子与基础奖励。两组辅助决策使用同样的触发频率、候选动作和 6 步预演，区别是排序方法。实际游戏执行入选动作序列的前 6 步，其余时间使用 PPO。

- 纯 PPO 采样
- PPO 加短预演规则排序
- PPO 加短预演小模型排序

## 复跑

在项目根目录安装训练依赖，并将交付包里的 `best_model.zip` 放到 `models/current/`。

```powershell
python train/train_decision_ranker.py --output logs/exp_20261103_ranker_01
```

输出权重 `decision-ranker.pt` 与逐局 `results.json`。可加载保存的权重再评测。

```powershell
python train/train_decision_ranker.py --ranker logs/exp_20261103_ranker_01/decision-ranker.pt --eval-seed 4242 --output logs/exp_20261103_ranker_holdout_01
```

预演使用当前锁定版本的 NES 私有快照接口，独立实例与主对局逐步比对位置、终止和旗标。结果用于同一关卡的可行性验证。模型使用绝对位置特征，跨关卡泛化需要单独验证。
