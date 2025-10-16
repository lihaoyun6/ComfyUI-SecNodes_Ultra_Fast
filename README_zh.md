# ComfyUI SeC Nodes Ultra Fast

这是一套为 **SeC (Segment Concept)** 设计的 ComfyUI 自定义节点  
采用了由 OpenIXCLab 开发的 SeC-4B 模型，这是目前顶尖的视频对象分割技术，性能超越 SAM 2.1  
> 关于安装和模型下载的教程，请向下滚动。

[\[📄 English\]](./README.md)

## 更新日志

### v1.2\_ultra\_fast (2025-10-16) 🎉 体验极致的显存控制与速度提升

**新特性:**

- **极低的显存需求**: 在 >= 3GB 显存的显卡上也能流畅运行 FP16 权重且无明显速度损失
- **无需磁盘缓存**: 不再使用磁盘作为帧缓存，从而保持处理过程完全在内存中进行
- **高速权重加载**: 权重加载速度提升10倍 (单文件加载速度从100秒提升至10秒)

### v1.1 (2025-10-13) - 单文件模型 & FP8 支持

**新特性:**

- **单文件模型格式**: 只需下载一个文件，告别分片的4文件格式
  - FP16 (7.35GB) - 推荐默认选项
  - ~~FP8 (3.97GB) - 适用于显存受限的系统 (需要 RTX 30系或更新的显卡)~~ (已弃用)
  - BF16 (7.35GB) - FP16 的替代方案
  - FP32 (14.14GB) - 全精度
- **FP8 量化支持**: 使用 torchao + Marlin 内核实现自动的仅权重(W8A16)量化
  - 在实际使用中可节省 1.5-2GB 显存
  - 需要 RTX 30系或更新的显卡 (Ampere架构及以上)
  - 在旧款 GPU 上会自动回退到 FP16
  
**变更:**

- 模型加载器现在支持多种精度格式并能自动检测。同时保留了对分片模型的兼容性。
- `requirements.txt` 中增加了 `torchao>=0.1.0` 以支持 FP8
- 自动检测 GPU 能力以判断是否兼容 FP8

**下载:** 新的单文件模型可在此处获取 [https://huggingface.co/VeryAladeen/Sec-4B](https://huggingface.co/VeryAladeen/Sec-4B/tree/main)

## 什么是 SeC?

**SeC (Segment Concept, 分割概念)** 是视频对象分割领域的一项突破性进展，它从简单的特征匹配转向了**高层次的概念理解**。与主要依赖视觉相似性的 SAM 2.1 不同，SeC 使用**大型视觉语言模型 (LVLM)** 来从概念上理解*什么*是对象，从而实现更稳健的追踪，具体优势包括：

- **语义理解**: 通过概念识别对象，而不仅仅是外观
- **场景复杂度自适应**: 自动平衡语义推理与特征匹配
- **卓越的稳健性**: 在处理遮挡、外观变化和复杂场景方面优于 SAM 2.1
- **SOTA 性能**: 在 SeCVOS 基准测试中比 SAM 2.1 高出 11.8 分

### SeC 的工作原理

1.  **视觉定位**: 你在某一帧上提供初始提示 (点/边界框/蒙版)
2.  **概念提取**: SeC 的 LVLM 分析对象以建立语义理解
3.  **智能追踪**: 动态地同时使用语义推理和视觉特征
4.  **关键帧库**: 维护对象的多个不同视角，以实现稳健的概念理解

最终结果？SeC 在快速外观变化、遮挡和复杂多对象场景等挑战性情况下，能更可靠地追踪对象。

## 演示

https://github.com/user-attachments/assets/5cc6677e-4a9d-4e55-801d-b92305a37725

*示例: SeC 在场景变化和动态移动中追踪对象*



https://github.com/user-attachments/assets/9e99d55c-ba8e-4041-985e-b95cbd6dd066

*示例: 在某些场景中，SAM 未能正确追踪到狗*

## 功能特性

- **SeC 模型加载器**: 通过简单的设置加载 SeC 模型
- **SeC 视频分割**: 使用视觉提示进行顶尖水平的视频分割
- **坐标绘制器**: 在分割前可视化坐标点
- **完全独立**: 所有推理代码都已打包 - 无需外部代码库
- **双向追踪**: 从任何帧向任何方向进行追踪

## 安装

### 方式一: ComfyUI-Manager (推荐 - 最简单)

1.  **安装 ComfyUI-Manager** (如果还没有的话):
    -   从此获取: https://github.com/ltdrdata/ComfyUI-Manager

2.  **下载一个模型** (见下面的模型下载部分)

3.  **安装 SeC 节点**:
    -   在 ComfyUI 中打开 ComfyUI Manager
    -   点击 "Install via Git URL"
    -   将 `https://github.com/lihaoyun6/ComfyUI-SecNodes_Ultra_Fast` 粘贴到文本框中
    -   点击 "确认" 进行安装

4.  **完成!** SeC 节点将出现在 "SeC" 类别下

---

### 方式二: 手动安装

#### 步骤 1: 安装自定义节点
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/9nate-drake/Comfyui-SecNodes
```

#### 步骤 2: 安装依赖项

**ComfyUI 便携版 (Windows):**

```bash
cd ComfyUI/custom_nodes/Comfyui-SecNodes
../../python_embeded/python.exe -m pip install -r requirements.txt
```

**标准 Python 环境 (Linux/Mac):**

```bash
cd ComfyUI/custom_nodes/Comfyui-SecNodes
pip install -r requirements.txt
```

#### 步骤 3: 重启 ComfyUI
节点将出现在 "SeC" 类别下。


## 模型下载

**根据你的显存/质量需求，下载以下任一模型格式:**

SeC 模型加载器会自动检测并让你选择使用哪个模型。  
从 [https://huggingface.co/VeryAladeen/Sec-4B](https://huggingface.co/VeryAladeen/Sec-4B) 下载并放置到你的 `ComfyUI/models/sams/` 文件夹中:

- 👍🏻**SeC-4B-fp16.safetensors** (推荐) - 7.35 GB
  - 质量和大小的最佳平衡
  - 适用于所有 CUDA GPU
  - 配合自动CPU卸载，可以运行在 3GB 可用显存 + 6GB 可用内存上运行
- ~~**SeC-4B-fp8.safetensors** - 3.97 GB~~ 
  - *已弃用*  
- **SeC-4B-bf16.safetensors** (备选) - 7.35 GB
  - FP16 的替代品，对某些 GPU 更友好
- **SeC-4B-fp32.safetensors** (全精度) - 14.14 GB
  - 最高精度，最高显存占用

---

#### 备选: 原始分片权重

**仅适用于喜欢 OpenIXCLab 原始格式的用户 🙅🏻‍♀️不推荐普通用户下载:**

```bash
cd ComfyUI/models/sams

# 使用 huggingface-cli 下载 (推荐)
huggingface-cli download OpenIXCLab/SeC-4B --local-dir SeC-4B

# 或者使用 git lfs
git lfs clone https://huggingface.co/OpenIXCLab/SeC-4B
```

**详情:**

- 大小: ~14.14 GB (分片为4个文件)
- 精度: FP32
- 下载内容包含所有配置文件

## 系统要求

- **Python**: 3.10-3.12 (推荐 3.12)
  - Python 3.13: 不推荐 - 处于实验性支持阶段，存在已知的依赖安装问题
- **PyTorch**: 2.6.0+ (ComfyUI 自带)
- **CUDA**: 11.8+ 用于 GPU 加速
- **CUDA GPU**: 推荐 (支持 CPU，但速度会慢很多)
- **显存**: 最低 3GB VRAM+ 6GB RAM 可用
    - 启用 `enable_cpu_offload` 可以动态控制显存占用, 配合手动显存限制可以进一步节省显存   
  - 启用 `offload_video_to_cpu` 可以显著降低显存占用 (速度约慢3%)

**关于 CPU 模式的说明:**

- CPU 推理会自动使用 float32 精度 (CPU 不支持 bfloat16/float16)
- 预计性能会远慢于 GPU (根据硬件不同，慢约10-20倍)
- 不建议用于生产环境，主要用于测试或无 GPU 的系统

**Flash Attention 2 (可选):**

- 提供约2倍的速度提升，但需要特定硬件
- **GPU 要求**: 仅限 Ampere/Ada/Hopper 架构 (RTX 30/40 系列, A100, H100)
  - 不适用于 RTX 20 系列 (Turing) 或更早的 GPU
- **CUDA**: 需要 12.0+
- **Windows + Python 3.12**: 使用预编译的 wheels 或禁用 flash attention
- 如果 Flash Attention 不可用，节点会自动回退到标准 attention

## 节点参考

### 1. SeC Model Loader (SeC 模型加载器)
加载并配置 SeC 模型以进行推理。自动检测 `ComfyUI/models/sams/` 目录中可用的模型。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| **model_file** | CHOICE | 第一个可用的模型 | 选择要加载的模型:<br>• FP32 (全精度 - ~14.5GB)<br>• FP16 (半精度 - 7.35GB)<br>• BF16 (Brain Float - ~7GB)<br>• FP8 (8位浮点 - 3.97GB)<br>• SeC-4B (分片/原始 - ~14GB)<br>**注意:** 每个模型都会自动使用其原生精度 |
| **device** | CHOICE | `auto` | 设备选择 (动态检测可用 GPU):<br>• `auto`: 如果有 cuda:0 则使用，否则使用 CPU (推荐)<br>• `cpu`: 强制使用 CPU (自动转换为 float32)<br>• `cuda:0`, `cuda:1`, 等: 指定 GPU |
| *use_flash_attn* | BOOLEAN | True | 启用 Flash Attention 2 以加速推理。<br>**注意:** 对 FP32/FP8 精度会自动禁用 (需要 FP16/BF16) |
| *enable_cpu_offload* | BOOLEAN | True | 自动卸载权重, 有助于节省显存 |
| *vram_limit* | INT | 0 | 显存占用上限, 0=自动分配 (单位为GB, 是参考值, 不绝对准确) |
| *allow_mask_overlap* | BOOLEAN | True | 允许对象重叠 (若要严格分离则禁用) |

**输出:** `model`

**说明:**

- **模型选择**: 动态显示 `ComfyUI/models/sams/` 目录中可用的模型
  - 至少下载一种模型格式 (见上方的模型下载部分)
  - 模型会以其**原生精度**加载 (FP8 保持为 FP8，不会向上转换!)
  - 这保留了较小模型格式的所有内存优势
- **配置文件**: 已捆绑在此仓库中 - 单文件模型无需单独下载
- **CPU 模式**: 自动将模型转换为 float32 精度 (CPU 限制)。CPU 推理比 GPU 慢得多 (~10-20倍)。
- **Flash Attention**: 对 FP32 和 FP8 模型自动禁用 (需要 FP16/BF16)。将改用标准 attention。

---

### 2. SeC Video Segmentation (SeC 视频分割)
在视频帧之间分割和追踪对象。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| **model** | MODEL | - | 从加载器输出的 SeC 模型 |
| **frames** | IMAGE | - | 作为 IMAGE 批次的视频帧 |
| *positive_points* | STRING | "" | JSON 格式: `'[{"x": 100, "y": 200}]'` |
| *negative_points* | STRING | "" | JSON 格式: `'[{"x": 50, "y": 50}]'` |
| *bbox* | STRING | "" | 边界框: `"x1,y1,x2,y2"` |
| *input_mask* | MASK | - | 二值蒙版输入 |
| *tracking_direction* | CHOICE | `forward` | forward / backward / bidirectional (前向/后向/双向) |
| *annotation_frame_idx* | INT | 0 | 应用提示的帧的索引 |
| *object_id* | INT | 1 | 用于多对象追踪的唯一 ID |
| *max_frames_to_track* | INT | -1 | 最大追踪帧数 (-1 = 全部) |
| *mllm_memory_size* | INT | 12 | 用于语义理解的关键帧数量 (影响场景变化时的计算量，不影响显存)。原始论文使用 7。 |
| *offload_video_to_cpu* | BOOLEAN | False | 将视频帧卸载到 CPU (显著节省 GPU 内存，速度慢约3%) |
| *auto_unload_model* | BOOLEAN | True | 分割后自动从显存和内存中卸载模型。如果需要连续进行多次分割，请设为 false。 |

**输出:** `masks` (MASK), `object_ids` (INT)

**重要说明:**

- 至少提供一个视觉提示 (点、边界框或蒙版)
- **输出的帧数始终与输入匹配**: 如果你输入100帧，你会得到100个蒙版
- 追踪范围之外的帧将是空的 (黑色) 蒙版
  - 示例: 100帧, annotation_frame_idx=50, direction=forward (前向) → 0-49帧为空，50-99帧被追踪
  - 示例: 100帧, annotation_frame_idx=50, direction=backward (后向) → 0-50帧被追踪，51-99帧为空
  - 示例: 100帧, annotation_frame_idx=50, direction=bidirectional (双向) → 所有 0-99 帧都被追踪

**输入组合行为:**

你可以组合不同的输入类型以实现强大的分割控制：

| 输入组合 | 行为 |
|-------------------|----------|
| **仅点** | 标准的基于点的分割 |
| **仅边界框** | 分割边界框内最突出的对象 |
| **仅蒙版** | 追踪蒙版区域 |
| **边界框 + 点** | **两阶段优化**: 边界框确定初始区域，然后点在该区域内优化分割 |
| **蒙版 + 正向点** | 只有**蒙版内部**的正向点会被用来优化要分割的蒙版区域部分 |
| **蒙版 + 负向点** | 所有负向点都用于从蒙版中排除区域 |
| **蒙版 + 正向 + 负向** | 蒙版内的正向点优化区域，负向点排除区域 |

**应用案例:**

- **边界框 + 点优化**: 在一个人周围画一个边界框，然后在他们的衬衫上加一个点，这样就只分割衬衫而不是整个人
- **粗略蒙版 + 精准点**: 在一个人周围画一个粗略的蒙版，然后在他们的脸上添加正向点以聚焦分割
- **蒙版 + 负向排除**: 蒙版一个对象，在不需要的部分添加负向点 (例如，从人物蒙版中排除一只手)
- **点过滤**: 蒙版边界外的正向点会被自动忽略，防止意外选择

**⚠ 关于蒙版与负向点的重要说明:**

- 负向点最好放置在蒙版区域**内部或附近**
- 离蒙版太远的负向点 (>50像素) 可能会导致意外结果或分割为空
- 如果负向点离蒙版太远，你会在控制台收到一条警告

---

### 3. Coordinate Plotter (坐标绘制器)
在图像上可视化坐标点以便调试。

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| **coordinates** | STRING | `'[{"x": 100, "y": 100}]'` | 要绘制的 JSON 坐标 |
| *image* | IMAGE | - | 可选图像 (会覆盖宽度/高度设置) |
| *point_shape* | CHOICE | `circle` | circle / square / triangle (圆形/方形/三角形) |
| *point_size* | INT | 10 | 点的大小，单位像素 (1-100) |
| *point_color* | STRING | `#00FF00` | 十六进制 `#FF0000` 或 RGB `255,0,0` |
| *width* | INT | 512 | 如果没有图像，画布的宽度 |
| *height* | INT | 512 | 如果没有图像，画布的高度 |

**输出:** `image` (IMAGE)


## 追踪方向

| 方向 | 最佳适用场景 | 行为 |
|-----------|----------|----------|
| **forward (前向)** | 标准视频，对象在开头出现 | 从第 N 帧 → 结尾 |
| **backward (后向)** | 对象在后面出现，反向分析 | 从第 N 帧 → 开头 |
| **bidirectional (双向)** | 对象在中间最清晰，复杂场景 | 从第 N 帧 → 双向进行 |


### 理解 mllm_memory_size

`mllm_memory_size` 参数控制 SeC 的大型视觉语言模型用于语义理解的历史关键帧数量：

- **作用**: 存储帧引用 (第一帧 + 最近的 N-1 帧)，供 LVLM 在场景变化时进行分析
- **显存影响**: 无 - 测试表明，3-20 的值使用相似的显存 (对于典型视频约 11-13GB)
- **计算影响**: 更高的值意味着在场景变化时有更多的帧通过视觉编码器处理
- **质量权衡**: 更多的关键帧 = 在复杂场景中对对象概念的理解更好，但超过约 10-12 帧后收益递减
- **原始研究**: SeC 论文使用了 7 并达到了 SOTA 性能 (比 SAM 2.1 高 11.8 分)，强调关键帧的“质量而非数量”

**推荐值:**

- **默认 (12)**: 平衡的方法 - 比论文的 7 提供更多上下文，但不过度
- **低 (5-7)**: 在简单视频上推理更快，与原始研究设置相符
- **高 (15-20)**: 为非常复杂的视频提供最大的语义上下文 (无显存惩罚)

**为什么它不影响显存？** 该参数存储的是轻量级的帧索引和蒙版数组，而不是完整的帧张量。当场景变化发生时，帧是按需从磁盘加载以供 LVLM 处理的。底层的 SAM2 架构最多支持 22 帧。

## 版权归属

此节点实现了由 OpenIXCLab 开发的 **SeC-4B** 模型。

- **模型仓库**: [OpenIXCLab/SeC-4B](https://huggingface.co/OpenIXCLab/SeC-4B)
- **论文**: [arXiv:2507.15852](https://arxiv.org/abs/2507.15852)
- **官方实现**: [github.com/OpenIXCLab/SeC](https://github.com/OpenIXCLab/SeC)
- **许可证**: Apache 2.0

**数据集**: 原始工作包括 [SeCVOS Benchmark](https://huggingface.co/datasets/OpenIXCLab/SeCVOS) 数据集。

## 已知局限性

**仅蒙版输入**: 只使用蒙版或边界框可能会导致追踪不够稳定。这是由于底层的 SAM2 和 MLLM 组件处理蒙版和边界框输入的方式所致。为获得最佳效果，请将蒙版/边界框与坐标点结合使用以实现更精确的控制。

## 故障排除


**CUDA out of memory (CUDA 显存不足)**:

- 启用 `enable_cpu_offload` 自动分配, 还可以手动设置显存阈值, 最低 3GB 即可运行
- 启用 `offload_video_to_cpu` (节省 2-3GB 显存，速度仅慢约3%)
- 同时确保你正在使用 fp8 变体以最大程度节省显存
- 一次处理更少的帧 (将视频分割成更小的批次)
- 查看上方针对你的硬件等级的 GPU 显存建议

**推理缓慢**:

- 在模型加载器中启用 `use_flash_attn` (需要 Flash Attention 2)
- 如果你有足够的显存，禁用 `offload_video_to_cpu`

---

*完全独立的 ComfyUI 节点 - 只需安装即可开始分割！* 🎉
