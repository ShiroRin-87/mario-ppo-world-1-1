# NES 环境安装

使用 64 位 Python 3.11。Windows 需要 Visual Studio 2022 的 C++ 桌面开发工作负载及 Windows SDK，供 pip 编译 `nes-py`。可先用 Visual Studio Installer 检查工作负载，再在 x64 Native Tools 命令行执行 `cl` 检查编译器。

在仓库根目录运行以下命令。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

冒烟脚本固定动作采样种子，对 World 1-1 执行 100 步随机动作，检查七动作空间、RGB 帧和距离信息。成功时输出 `PASS steps=100 actions=7` 和实测数字。

## WSL2 备选环境

在已启用的 WSL2 Linux 发行版中使用 Python 3.11，安装 `build-essential`、`clang` 与对应的 Python 头文件和 venv 组件。重新创建 Linux venv，再按相同顺序运行。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/smoke_test.py
```

WSL2 上的图形窗口依赖 WSLg。冒烟读取 RGB 数组即可。

## 故障记录

| 现象 | 处理 |
|---|---|
| 系统默认 Python 3.12 | 在独立目录部署 Python 3.11 |
| PyPI 下载间歇出现 `UNEXPECTED_EOF_WHILE_READING` | 重试下载并用缓存的官方 wheel 验证，保留版本约束 |
| 新版 Mario 后端要求 Python ≥3.13 | 采用 7.4 系列旧 Gym 后端，由项目环境封装暴露 Gymnasium 接口 |
| Windows 出现 C++ 编译器缺失 | 安装或补齐 MSVC 和 Windows SDK，再重试安装 |
