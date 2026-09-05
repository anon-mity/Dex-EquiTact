# Dex-EquiTact

[English](README.md) | 简体中文

根据 Dex-EquiTact 方法重新实现的独立 PyTorch 仓库，以视觉、本体状态和三轴指尖力
驱动灵巧手反应式策略。包含双类型等变触觉编码、因果扩散动作策略、EMA 未来力潜变量
监督、可恢复训练，以及逐触觉帧流式推理。

当前版本已使用合成输入完成软件验证，**尚未用原始示教数据训练，也未复现论文的
机器人成功率**。原始数据、训练好的检查点和机器人驱动不包含在仓库中。

## 方法概览

- **保留类型与五指顺序。** 腕坐标系中的指尖位置和经过标定的传感器局部力分别进入
  因果 VN 编码器，遵循两个独立的 SO(3) 旋转规律。表示保持为
  `[B,T,5,2,Cv,3]` 向量场，五指顺序固定为拇指、食指、中指、无名指、小指。
- **共享视觉调制。** 两个慢速 RGB/本体观测生成缓存 token 和标量 FiLM 增益，FiLM
  调制后的共享表示仍为向量场。动作专家内部才展平触觉，同时直接读取慢观测 token。
  等变性描述针对双类型触觉表示，不扩展为完整动作策略的等变性保证。
- **因果反应式扩散。** 动作专家预测注入噪声 epsilon，动作之间及动作对触觉的注意力
  均按时间施加因果 mask。每次触觉到达，确定性 DDIM（`eta=0`）从缓存的原始噪声和
  慢上下文重新采样已到达的动作前缀，只返回最新动作供调用方执行。
- **未来力潜变量监督。** 16 个当前触觉时刻各预测随后 8 步力潜变量。预测头使用共享
  FiLM 表示，force-only EMA 教师提供不回传梯度的目标，在每次优化器更新后更新。
  预测头和教师只用于训练，均不需要动作输入。

论文未指定的 CNN、网络宽度、优化器、扩散日程等采用明确列出的工程默认值。
架构图见[英文主页](README.md#method-overview)，详细对应关系和来源记录见
[METHOD_ALIGNMENT.md](docs/METHOD_ALIGNMENT.md) 与
[IMPLEMENTATION_CONTRACT.md](docs/IMPLEMENTATION_CONTRACT.md)。

## 安装与快速验证

需要 Python >=3.10、PyTorch >=2.2、NumPy 和 PyYAML。CNN 从头训练，安装不下载
预训练权重：

```bash
git clone https://github.com/anon-mity/Dex-EquiTact.git
cd Dex-EquiTact
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m dex_equitact smoke --output /tmp/dex_smoke_wuji --action-dim 26
python -m dex_equitact smoke --output /tmp/dex_smoke_sharpa --action-dim 28
```

Smoke 创建明确标记的合成数据，用小型测试网络执行两次真实 forward/backward、
优化和 EMA 更新、检查点保存/加载，以及完整 16 帧流式推理。动作输出形状分别为
`[1,16,26]` 和 `[1,16,28]`，结果写入 `smoke_result.json`。输出目录必须为空或不存在。

安装后也可使用 `dex-equitact` 命令。未安装包时，在仓库根目录使用
`PYTHONPATH=src python -m dex_equitact ...`。
`python -m dex_equitact train --help` 可查看训练参数。

## 机器人配置

分别训练 26 维和 28 维策略：

| 配置 | 文件 | 动作布局 | 本体状态维度 |
| --- | --- | --- | --- |
| WUJI | [wuji.yaml](configs/wuji.yaml) | 6 维机械臂增量 + 20 维手部目标 = 26 | 26 |
| Sharpa | [sharpa.yaml](configs/sharpa.yaml) | 6 维机械臂增量 + 22 维手部目标 = 28 | 28 |

两份配置均使用两个相机、两个慢观测、16 个触觉/动作时刻和 8 步未来力目标。
默认向量通道数为 32，动作 Transformer 宽度为 128，训练扩散级数为 1000，
DDIM 推理更新次数为 10。这些网络与扩散参数是工程默认值。

## 准备数据和训练

格式见 [DATA_FORMAT.md](docs/DATA_FORMAT.md)。每条示教为一个 NPZ，包含 `images`、
`proprio`、`positions`、`forces`、`actions`、`timestamps` 和 `image_timestamps`。
`manifest.json` 显式列出 episode，并记录五指顺序、位置/力坐标系、单位、
标定来源和完整动作约定。位置须为实测或独立验证的腕坐标系指尖位置，单位米；
力须为经过标定的传感器局部三维力，单位牛顿。真实几何、时间戳及绝对目标到增量
动作的转换由数据生产方提供。可选 [Zarr 转换器](scripts/convert_zarr.py) 的用法和
前提也见数据格式文档。

训练/验证按完整 episode 划分，归一化只使用训练集。位置和力各使用一个在 XYZ、
五指和时间上共享的标量，不加向量偏移；动作和本体状态使用逐分量均值/标准差。
训练需要独立且可用的训练/验证 episode；默认窗口至少需要 40 个已对齐帧。
`--slow-stride` 默认 16，表示两个慢观测之间相隔的已对齐快采样帧数，应按采集协议
设置；这个数值不代表实际采样频率。

```bash
python -m dex_equitact validate-data --data /path/to/dataset

python -m dex_equitact train --config configs/wuji.yaml \
  --data /path/to/dataset --output runs/wuji --device cuda \
  --steps 100000 --batch-size 8 --save-every 1000

python -m dex_equitact train --config configs/wuji.yaml \
  --data /path/to/dataset --output runs/wuji --device cuda \
  --steps 200000 --batch-size 8 --save-every 1000 --resume runs/wuji/last.pt

python -m dex_equitact evaluate --data /path/to/dataset \
  --checkpoint runs/wuji/last.pt --device cuda
```

将 `/path/to/dataset` 替换为实际数据路径。CPU 可使用 `--device cpu`；CUDA 需要可用
的对应安装。示例训练步数是用法示例，不代表论文实验设置。Sharpa 使用
`configs/sharpa.yaml` 和独立的输出目录。

继续训练时指定相同配置、数据和已记录的训练设置，`--steps` 表示**目标总优化步数**，
包含已完成的步数。`last.pt` 保存 online/EMA、优化器、随机状态、采样随机状态、
归一化统计、数据划分、配置和数据内容哈希。每次保存同时记录验证损失；
`metrics.jsonl` 和 `run.json` 记录运行信息。数据或关键训练设置变化会拒绝原样续训。
评估使用检查点记录的留出集，输出动作去噪和潜变量损失；机器人任务成功率需另行评估。

## 流式推理接口

下面假定观测已经采集。输入使用与策略位于同一 device 的 float32 tensor。
`B` 为批量大小，`V` 为相机数，`P` 为本体状态维度，后两者须匹配检查点；
图像高宽 `H,W` 均须至少为 8。

```python
from dex_equitact.training import load_policy
from dex_equitact.streaming import ReactiveController

policy, normalizer, checkpoint = load_policy("runs/wuji/last.pt", device="cpu")
controller = ReactiveController(policy, normalizer)  # 加载后已处于 eval 模式

# 已经采集到的两个慢观测；RGB 为 [0,1]，本体状态保持记录中的物理单位。
# images: [B,2,V,3,H,W]; proprio: [B,2,P]
controller.start_cycle(images, proprio, timestamp=visual_time)

# 按约定五指顺序，每次只输入新到达的触觉；positions/forces: [B,5,3]。
# 位置为 wrist-frame 米，力为各指 sensor-local 牛顿。
latest_action = controller.step(positions, forces, timestamp=tactile_time)
# latest_action: [B,26] 或 [B,28]，已反归一化；由调用方执行这一条。
```

传入检查点 normalizer 时，输入本体状态和触觉使用物理单位，输出动作为记录约定下
的物理量。使用 `ReactiveController(policy)` 不传 normalizer 时，所有非图像输入/
输出均视为**已归一化值**；RGB 始终为 `[0,1]`。

时间戳使用有限的秒数。每个触觉时间戳须不早于当前周期开始，并严格晚于上一触觉
时间戳。一个视觉周期最多 16 次 `step`；新慢观测到达时再次调用 `start_cycle`，
刷新上下文、清空触觉历史并缓存新噪声。调用方负责硬件连接，并按照 manifest 中
声明的平移坐标系、旋转组合规则和手关节单位解释动作。

## 验证

```bash
python -m pytest -q
```

[VALIDATION.md](docs/VALIDATION.md) 记录了 Python 3.12 / PyTorch 2.11 环境中
**60 项 CPU 测试通过**，覆盖双类型旋转、五指顺序、因果性、DDIM 前缀一致性、
EMA 目标、两种动作维度的流式推理、数据约定和检查点恢复。这些测试使用合成输入；
CUDA 执行和真实机器人时延/性能仍未验证。

## 方法与代码对应

| 内容 | 文件 |
| --- | --- |
| 双流因果 VN、保留五指 | [vn.py](src/dex_equitact/models/vn.py), [tactile.py](src/dex_equitact/models/tactile.py) |
| 缓存慢图像/本体 token | [vision.py](src/dex_equitact/models/vision.py) |
| 标量 FiLM、双类型未来力预测头 | [tactile.py](src/dex_equitact/models/tactile.py) |
| 双因果 mask、epsilon prediction、DDIM | [diffusion.py](src/dex_equitact/models/diffusion.py) |
| 共享表示、联合损失、force-only EMA | [policy.py](src/dex_equitact/policy.py) |
| 固定噪声/慢上下文、最新动作输出 | [streaming.py](src/dex_equitact/streaming.py) |
| 数据、归一化、检查点恢复 | [data.py](src/dex_equitact/data.py), [training.py](src/dex_equitact/training.py) |
| 配置、CLI 与测试 | [config.py](src/dex_equitact/config.py), [cli.py](src/dex_equitact/cli.py), [tests/](tests) |

## 许可

采用 [MIT License](LICENSE)。
