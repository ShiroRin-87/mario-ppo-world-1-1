# World 1-1 十路同屏演示

左侧 4 个大画面、右侧 6 个小画面同时展示十个独立的 NES 实例。画面按 16:15 原始比例铺满 1920×1080 画布，标题和状态叠在画面边缘。实例分别覆盖标准开局、四种开局输入、中途松手、反向、跳跃、多次覆盖及死亡后的检查点恢复。每路按自己的模拟器状态实时决策，错峰启动方便观察。各格显示当前横向位置、覆盖事件和旗标结果。

每格底部的「键」显示生成当前游戏帧时实际送入模拟器的按键。方向用箭头表示，A 为跳跃，B 为加速，松开表示无按键。橙色按键来自注入扰动。「—」表示该画面对应的时刻没有执行新指令，例如等待开局、恢复检查点或结束后定格时。

## 使用

将上一次交付的 `world-1-1-route.npz` 放到 `models/current/`。在项目根目录运行。

```powershell
python viz/ten_screen_demo.py --route-model models/current/world-1-1-route.npz --preview
```

预览使用 Tk 窗口，按 F11 切换全屏，按 Esc 退出。保存同屏录像时使用 FFmpeg。

```powershell
python viz/ten_screen_demo.py --route-model models/current/world-1-1-route.npz --output data/ten-screen.mp4
```

默认使用 Windows 的微软雅黑字体。可用 `--font` 指定其他中文字体。MP4 为 1920×1080、15 FPS，配套 JSON 保存十个实例的通关状态、输入覆盖与检查点恢复。画面显示的是模拟器真实帧。
