# Football Video Action Spotting Project

Football Video Action Spotting Project

本项目基于 SoccerNet 官方发布的 SoccerNet-v2 数据集及其公开基线代码实现，在 CALF（Context-Aware Loss Function）框架基础上进行了功能扩展、结构调整与工程化改进。

相较于原始实现，本项目主要完成了以下工作：

对训练流程与视频推理流程进行了拆分与重构，降低模块间耦合度；
调整训练入口，使其更适合在 VS Code 等 IDE 中直接调试与运行；
整理并优化完整视频推理流程，支持从输入视频到事件预测结果的端到端处理；
补充自动化处理脚本与工程目录结构，便于实验复现与后续开发；
保留与原始 CALF 框架兼容的训练方式与模型结构，便于与官方基线结果进行比较。

需要说明的是，本项目并非 SoccerNet 官方仓库的镜像版本。部分代码来源于 SoccerNet 官方公开实现，并在此基础上根据研究需求进行了修改与扩展。若需要了解原始实现或获取最新版本代码，请参考 SoccerNet 官方仓库。

若需要使用 FFmpeg / FFmpy，请自行安装并确保其可在系统 PATH 中被正确访问。

---

## 1. 项目定位

该项目包含两条主线：

1. 训练流程：用于在 SoccerNet 数据集上训练/验证/测试模型。
2. 全视频推理流程：用于对单个外部视频进行完整推理，输出动作预测结果。

当前版本的核心原则是：
- 训练与推理尽量分离，避免互相耦合。
- 训练入口建议在 VS Code / 编辑器中直接运行，便于调试与参数调整。
- 如果调整了特征提取方式，训练和推理必须使用完全一致的特征提取流程。

---

## 2. 重要约束（必须遵守）

### 2.1 特征提取必须一致

如果你修改了特征提取方式（例如 ResNET / PCA / 帧采样 / 裁剪方式 / FPS 等），请务必保证：
- 训练阶段使用的特征提取方式
- 推理阶段使用的特征提取方式

必须保持一致。

这包括但不限于：
- TensorFlow 版本
- PyTorch 版本
- 特征提取脚本与参数
- PCA 文件与均值文件
- 帧采样率 / 裁剪策略 / 输入尺寸

### 2.2 TensorFlow / PyTorch 版本要严格保持一致

为了保证复现性与结果稳定，建议使用以下版本组合：

- Python：3.8
- PyTorch：1.6
- torchvision：0.7
- CUDA Toolkit：10.1（如果使用 GPU）
- TensorFlow：2.3.0
- OpenCV：3.4.11.41

如果你更换了特征提取实现，务必同时检查：
- 训练环境和推理环境是否使用同一套 TensorFlow / PyTorch 版本
- 是否使用了同一套 PCA / ResNET 特征文件

### 2.3 FFmpeg / FFmpy 需要自行安装

本项目中涉及视频处理与特征提取时，依赖 FFmpeg / FFmpy。

请自行完成以下准备：
1. 安装 FFmpeg 并把可执行文件路径加入系统 PATH(bin)。
2. 安装 Python 包：ffmpy

如果 FFmpeg 不在 PATH 中，视频处理或特征提取可能出现失败。

---

## 3. 推荐环境

建议使用 Conda 创建独立环境：

```bash
conda create -n footballvideo python=3.8
conda activate footballvideo
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn ffmpy
```

如果你的机器没有 GPU，也可以先在 CPU 环境下验证，但推荐保持版本一致。

---

## 4. 项目结构

```text
.
├── Full_process/          # 训练与完整流程代码
│   ├── src/               # 训练主代码
│   ├── inference/         # 外部视频推理流程
│   ├── models/            # 模型权重与结果
│   └── outputs/           # 预测输出
├── inference/             # 推理相关代码（可选复用版本）
├── models/                # 预训练模型与日志
└── README.md              # 本说明文件
```

---

## 5. 训练流程

### 5.1 训练入口

训练代码主要位于：
- Full_process/src/main.py

建议做法：
- 在 VS Code 中打开 Full_process/src/main.py
- 直接用编辑器的 Run / Debug 方式启动
- 根据需要在参数配置中调整数据路径、特征文件、模型名、批大小等

说明：
- 这套流程已经从命令行执行方式改成了更适合编辑器调试的方式。
- 若要做模型实验，建议在编辑器中直接设置参数并运行。

### 5.2 训练时的注意事项

- 训练时使用的特征文件必须与推理时一致。
- 若更换了特征提取脚本或调整了参数，请同步检查推理侧。
- 如果你想比较不同模型，请保留一套固定的特征文件和版本配置，以便结果可复现。

---

## 6. 视频全推理流程

推理代码主要位于：
- Full_process/inference/
- inference/

推理流程一般包括：
1. 输入视频
2. 视频预处理 / 转码
3. 特征提取
4. PCA 处理（如果使用）
5. 模型预测
6. 输出 JSON / 可视化结果

### 6.1 推理时的必要要求

- 必须使用与训练阶段相同的特征提取方式。
- 必须使用与训练阶段一致的 TensorFlow / PyTorch 版本。
- 如果使用 FFmpeg，请保证其可执行文件在 PATH 中。

---

## 7. 如果你准备调整特征提取方式

当你需要调整数据特征提取方式时，请按下面的流程处理：

1. 先确定新方案的完整参数集合。
2. 同时更新训练侧与推理侧的特征提取逻辑。
3. 保证两边使用同一套：
   - 特征文件
   - PCA 文件
   - 提取脚本
   - TensorFlow / PyTorch 版本
   - 帧采样策略与输入尺寸

> 只要训练和推理的特征提取流程不一致，结果就可能出现明显偏差，因此请不要只改一边。

---

## 8. 推荐使用方式

### 方式 A：训练与调试
- 在 VS Code 里打开 Full_process/src/main.py
- 直接运行调试
- 观察日志、模型权重与输出结果

### 方式 B：视频推理
- 打开 Full_process/inference/ 相关入口
- 输入待预测视频路径
- 运行后输出预测 JSON 与相关结果文件

---

## 9. 结论

这套项目现在的重点是：
- 训练和推理流程分离
- 训练入口适合编辑器运行
- FFmpeg / FFmpy 需要用户自行安装
- 特征提取方式必须严格保持一致，尤其是 TensorFlow 与 PyTorch 版本

如果后续要继续开发或复现实验，请优先保持“训练与推理完全一致”的原则。
## 项目来源与致谢

本项目部分代码基于 SoccerNet 官方公开实现修改而来，相关数据集、评测协议及基线方法请参考 SoccerNet 官方资源。

如果本项目被用于学术研究，请同时引用：

SoccerNet-v2 数据集相关论文；
CALF（Context-Aware Loss Function）相关论文；
SoccerNet 官方代码仓库及其文档。

本仓库中的修改内容主要涉及训练流程整理、推理流程重构、工程化部署以及实验相关改进，不代表 SoccerNet 官方实现。
