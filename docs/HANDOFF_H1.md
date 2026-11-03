# H1 环境交接

- 交出岗位：AS-1
- 接收岗位：IM-1
- 接收日期：2026-11-03
- Git 标签：`h1-env-ok`
- 上游提交：`05e6e53038d46f6c999da4c17177af3159f911c6`

## 1 能跑的命令

Windows 64 位 Python 3.11，在新克隆的仓库根目录运行。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

实测 Python 3.11.18、Gym 0.26.2、gym-super-mario-bros 7.4.0、nes-py 8.2.1。使用当前 Visual Studio 2022 工具链构建 NES wheel，在新虚拟环境从缓存 wheel 安装。在独立克隆目录复跑得到 `No broken requirements found.` 和 `PASS steps=100 actions=7 max_x_pos=155 total_reward=110.0`。

## 2 已知问题

- 默认 Python 为 3.12，本轮部署了隔离的 3.11
- PyPI 出现过 TLS 连接提前关闭，重试并使用本地构建的 wheel 完成复跑
- v0 会触发旧 Gym 提示，该关卡 ID 按规范保留
- WSL2 方案已写明步骤，本轮验证 Windows

## 3 接手入口

从 `scripts/smoke_test.py` 查看后端调用与七动作映射。在 `envs/` 实现 Gymnasium 适配，保留 AS-1 冒烟文件的独立性。

## 4 接收记录

- [x] 在独立克隆目录复跑并检查输出
- [x] 接收检查未修改上游文件

本次由同一执行者分别模拟 AS-1 和 IM-1，在两个独立目录执行。
