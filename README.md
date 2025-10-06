# ComfyUI SeC Nodes

**Self-contained** ComfyUI custom nodes for **SeC (Segment Concept)** - State-of-the-art video object segmentation that outperforms SAM 2.1.

## What is SeC?

**SeC (Segment Concept)** is a breakthrough in video object segmentation that shifts from simple feature matching to **high-level conceptual understanding**. Unlike SAM 2.1 which relies primarily on visual similarity, SeC uses a **Large Vision-Language Model (LVLM)** to understand *what* an object is conceptually, enabling robust tracking through:

- 🧠 **Semantic Understanding**: Recognizes objects by concept, not just appearance
- 🎯 **Scene Complexity Adaptation**: Automatically balances semantic reasoning vs feature matching
- 💪 **Superior Robustness**: Handles occlusions, appearance changes, and complex scenes better than SAM 2.1
- 📊 **SOTA Performance**: +11.8 points over SAM 2.1 on SeCVOS benchmark

### How SeC Works

1. **Visual Grounding**: You provide initial prompts (points/bbox/mask) on one frame
2. **Concept Extraction**: SeC's LVLM analyzes the object to build a semantic understanding
3. **Smart Tracking**: Dynamically uses both semantic reasoning and visual features
4. **Keyframe Bank**: Maintains diverse views of the object for robust concept understanding

The result? SeC tracks objects more reliably through challenging scenarios like rapid appearance changes, occlusions, and complex multi-object scenes.

## Features

- 🔥 **SeC Model Loader**: Load SeC models with simple settings
- 🎯 **SeC Video Segmentation**: SOTA video segmentation with visual prompts
- 🎨 **Coordinate Plotter**: Visualize coordinate points before segmentation
- 🚀 **Self-Contained**: All inference code bundled - no external repos needed
- ⚡ **Bidirectional Tracking**: Track from any frame in any direction

## Installation

### 1. Install Custom Node
Copy the `comfyui_sec_nodes` folder to your ComfyUI custom_nodes directory:

**Windows Portable:**
```
ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_sec_nodes\
```

**Standard Install:**
```
ComfyUI\custom_nodes\comfyui_sec_nodes\
```

### 2. Install Dependencies
```bash
cd ComfyUI/custom_nodes/comfyui_sec_nodes
pip install -r requirements.txt
```

### 3. Download SeC Model to ComfyUI Models Folder
The model should be placed in the standard ComfyUI models location:

**Target Location:**
```
ComfyUI/models/sams/SeC-4B/
```

**Download using huggingface-cli (recommended):**
```bash
# Navigate to ComfyUI models/sams folder
cd ComfyUI/models/sams

# Download model
huggingface-cli download OpenIXCLab/SeC-4B --local-dir SeC-4B
```

**Or using git lfs:**
```bash
cd ComfyUI/models/sams
git lfs clone https://huggingface.co/OpenIXCLab/SeC-4B
```

The node will automatically find the model at `models/sams/SeC-4B` (default path).

### 4. Restart ComfyUI
The nodes will appear in the "SeC" category.

## Nodes Reference

### 1. SeC Model Loader
Load and configure the SeC model for inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **model_path** | STRING | `models/sams/SeC-4B` | Path to model or HuggingFace ID |
| **torch_dtype** | CHOICE | `bfloat16` | Precision: bfloat16 (recommended), float16, float32 |
| **device** | CHOICE | `auto` | Device: auto, cuda, cpu |
| *use_flash_attn* | BOOLEAN | True | Enable Flash Attention 2 for faster inference |
| *allow_mask_overlap* | BOOLEAN | True | Allow objects to overlap (disable for strict separation) |

**Outputs:** `model`, `tokenizer`

---

### 2. SeC Video Segmentation
Segment and track objects across video frames.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **model** | MODEL | - | SeC model from loader |
| **tokenizer** | TOKENIZER | - | SeC tokenizer from loader |
| **frames** | IMAGE | - | Video frames as IMAGE batch |
| *positive_points* | STRING | "" | JSON: `'[{"x": 100, "y": 200}]'` |
| *negative_points* | STRING | "" | JSON: `'[{"x": 50, "y": 50}]'` |
| *bbox* | STRING | "" | Bounding box: `"x1,y1,x2,y2"` |
| *input_mask* | MASK | - | Binary mask input |
| *tracking_direction* | CHOICE | `forward` | forward / backward / bidirectional |
| *annotation_frame_idx* | INT | 0 | Frame where prompt is applied |
| *object_id* | INT | 1 | Unique ID for multi-object tracking |
| *max_frames_to_track* | INT | -1 | Max frames (-1 = all) |
| *mllm_memory_size* | INT | 7 | Semantic memory size |

**Outputs:** `masks` (MASK), `object_ids` (INT)

**Note:** Provide at least one visual prompt (points, bbox, or mask).

---

### 3. Coordinate Plotter
Visualize coordinate points on images for debugging.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **coordinates** | STRING | `'[{"x": 100, "y": 100}]'` | JSON coordinates to plot |
| *image* | IMAGE | - | Optional image (overrides width/height) |
| *point_shape* | CHOICE | `circle` | circle / square / triangle |
| *point_size* | INT | 10 | Point size in pixels (1-100) |
| *point_color* | STRING | `#00FF00` | Hex `#FF0000` or RGB `255,0,0` |
| *width* | INT | 512 | Canvas width if no image |
| *height* | INT | 512 | Canvas height if no image |

**Outputs:** `image` (IMAGE)

## Quick Start Examples

### Basic Workflow
```
1. SeC Model Loader (default settings)
   └─→ model, tokenizer

2. Load Video Frames
   └─→ frames

3. SeC Video Segmentation
   ├─ model: from (1)
   ├─ tokenizer: from (1)
   ├─ frames: from (2)
   └─ positive_points: '[{"x": 200, "y": 300}]'

   └─→ masks (ready for VideoCombine, etc.)
```

### With Coordinate Visualization
```
1. Load Image (first frame)
   └─→ image

2. Coordinate Plotter
   ├─ coordinates: '[{"x": 200, "y": 300}]'
   ├─ image: from (1)
   └─ point_color: "#FF0000"

   └─→ Preview image (verify point placement)

3. Use same coordinates in SeC Video Segmentation
```

### Bidirectional Tracking (Best for Complex Videos)
```
SeC Video Segmentation:
  └─ annotation_frame_idx: 25  (clear frame in middle)
  └─ tracking_direction: bidirectional
  └─ positive_points: '[{"x": 300, "y": 200}]'

Result: Tracks from frame 25 → forward to end AND backward to start
```

## Tracking Directions

| Direction | Best For | Behavior |
|-----------|----------|----------|
| **forward** | Standard videos, object appears at start | Frame N → end |
| **backward** | Object appears later, reverse analysis | Frame N → start |
| **bidirectional** | Object clearest in middle, complex scenes | Frame N → both directions |

## Performance Comparison

| Model | DAVIS 2017 | MOSE | SA-V | SeCVOS |
|-------|------------|------|------|--------|
| SAM 2.1 | 90.6 | 74.5 | 78.6 | **58.2** |
| SAM2Long | 91.4 | 75.2 | 81.1 | 62.3 |
| **SeC** | **91.3** | **75.3** | **82.7** | **70.0** |

SeC achieves **+11.8 points** over SAM 2.1 on complex semantic scenarios (SeCVOS).

## Requirements

- **Python**: 3.10-3.12
- **PyTorch**: Included with ComfyUI
- **CUDA GPU**: Recommended (CPU supported but slow)
- **VRAM**: ~16GB for SeC-4B model with bfloat16

## Links & Resources

- 📄 **Paper**: [arXiv:2507.15852](https://arxiv.org/abs/2507.15852)
- 🤗 **Model**: [OpenIXCLab/SeC-4B](https://huggingface.co/OpenIXCLab/SeC-4B)
- 📊 **Dataset**: [SeCVOS Benchmark](https://huggingface.co/datasets/OpenIXCLab/SeCVOS)
- 💻 **Original Repo**: [GitHub - SeC](https://github.com/OpenIXCLab/SeC)

## Troubleshooting

**Model not found**: Ensure model is at `ComfyUI/models/sams/SeC-4B/`

**CUDA out of memory**: Try `float16` or reduce `mllm_memory_size`

**Slow inference**: Enable `use_flash_attn` (requires Flash Attention 2)

**Empty masks**: Provide clearer visual prompts or try different frame

---

*Self-contained ComfyUI nodes - just install and segment!* 🎉