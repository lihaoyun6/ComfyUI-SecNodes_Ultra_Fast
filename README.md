# ComfyUI SeC Nodes

**Self-contained** ComfyUI custom nodes for **SeC (Segment Concept)** - State-of-the-art video object segmentation that outperforms SAM 2.1.

## What is SeC?

**SeC (Segment Concept)** is a breakthrough in video object segmentation that shifts from simple feature matching to **high-level conceptual understanding**. Unlike SAM 2.1 which relies primarily on visual similarity, SeC uses a **Large Vision-Language Model (LVLM)** to understand *what* an object is conceptually, enabling robust tracking through:

- **Semantic Understanding**: Recognizes objects by concept, not just appearance
- **Scene Complexity Adaptation**: Automatically balances semantic reasoning vs feature matching
- **Superior Robustness**: Handles occlusions, appearance changes, and complex scenes better than SAM 2.1
- **SOTA Performance**: +11.8 points over SAM 2.1 on SeCVOS benchmark

### How SeC Works

1. **Visual Grounding**: You provide initial prompts (points/bbox/mask) on one frame
2. **Concept Extraction**: SeC's LVLM analyzes the object to build a semantic understanding
3. **Smart Tracking**: Dynamically uses both semantic reasoning and visual features
4. **Keyframe Bank**: Maintains diverse views of the object for robust concept understanding

The result? SeC tracks objects more reliably through challenging scenarios like rapid appearance changes, occlusions, and complex multi-object scenes.

## Features

- **SeC Model Loader**: Load SeC models with simple settings
- **SeC Video Segmentation**: SOTA video segmentation with visual prompts
- **Coordinate Plotter**: Visualize coordinate points before segmentation
- **Self-Contained**: All inference code bundled - no external repos needed
- **Bidirectional Tracking**: Track from any frame in any direction

## Installation

### 1. Install Custom Node
```
cd \*installdirectory*\ComfyUI\custom_nodes\
git clone https://github.com/9nate-drake/Comfyui-SecNodes
```

### 2. Install Dependencies
```bash
cd ComfyUI/custom_nodes/Comfyui-SecNodes
pip install -r requirements.txt
```

### 3. Restart ComfyUI
The nodes will appear in the "SeC" category.

### 4. Model Auto-Download
**The SeC-4B model will automatically download on first use!** No manual download required.

When you first use the SeC Model Loader node, it will:
1. Check for existing model in `ComfyUI/models/sams/SeC-4B/`
2. If not found, automatically download from HuggingFace (~8.5GB)
3. Save to `ComfyUI/models/sams/SeC-4B/` for future use

**Optional: Pre-download Model (Faster First Run)**
```bash
# Navigate to ComfyUI models directory
cd ComfyUI/models/sams

# Download model using huggingface-cli
huggingface-cli download OpenIXCLab/SeC-4B --local-dir SeC-4B

# Or using git lfs
git lfs clone https://huggingface.co/OpenIXCLab/SeC-4B
```

## Nodes Reference

### 1. SeC Model Loader
Load and configure the SeC model for inference. Automatically downloads SeC-4B model on first use.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **torch_dtype** | CHOICE | `bfloat16` | Precision: bfloat16 (recommended for GPU), float16, float32.<br>**Note:** CPU mode automatically uses float32 regardless of selection |
| **device** | CHOICE | `auto` | Device selection (dynamically detects available GPUs):<br>• `auto`: gpu0 if available, else CPU (recommended)<br>• `cpu`: Force CPU (automatically uses float32)<br>• `gpu0`, `gpu1`, etc.: Specific GPU |
| *use_flash_attn* | BOOLEAN | True | Enable Flash Attention 2 for faster inference.<br>**Note:** Automatically disabled for float32 precision (requires float16/bfloat16) |
| *allow_mask_overlap* | BOOLEAN | True | Allow objects to overlap (disable for strict separation) |

**Outputs:** `model`

**Notes:**
- The model is automatically located in `models/sams/SeC-4B/` or downloaded from HuggingFace if not found.
- **Device options dynamically adapt** to your system:
  - 1 GPU system: Shows `auto`, `cpu`, `gpu0`
  - 2 GPU system: Shows `auto`, `cpu`, `gpu0`, `gpu1`
  - 3+ GPU system: Shows all available GPUs
  - No GPU: Shows only `auto` and `cpu`
- **CPU mode**: Automatically overrides to float32 precision to avoid dtype mismatch errors. CPU inference is significantly slower than GPU.
- **Float32 precision**: Flash Attention is automatically disabled when using float32 (requires float16/bfloat16). Standard attention will be used instead (slower but compatible).
- Dtype conversion hooks are automatically installed for GPU modes to ensure proper precision handling
- Model is automatically unloaded from memory after workflow completes (relies on Python garbage collection)

---

### 2. SeC Video Segmentation
Segment and track objects across video frames.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **model** | MODEL | - | SeC model from loader |
| **frames** | IMAGE | - | Video frames as IMAGE batch |
| *positive_points* | STRING | "" | JSON: `'[{"x": 100, "y": 200}]'` |
| *negative_points* | STRING | "" | JSON: `'[{"x": 50, "y": 50}]'` |
| *bbox* | STRING | "" | Bounding box: `"x1,y1,x2,y2"` |
| *input_mask* | MASK | - | Binary mask input |
| *tracking_direction* | CHOICE | `forward` | forward / backward / bidirectional |
| *annotation_frame_idx* | INT | 0 | Frame where prompt is applied |
| *object_id* | INT | 1 | Unique ID for multi-object tracking |
| *max_frames_to_track* | INT | -1 | Max frames (-1 = all) |
| *mllm_memory_size* | INT | 20 | Number of keyframes stored for semantic understanding (affects compute on scene changes, not VRAM) |
| *offload_video_to_cpu* | BOOLEAN | False | Offload video frames to CPU (saves significant GPU memory, ~3% slower) |

**Outputs:** `masks` (MASK), `object_ids` (INT)

**Important Notes:**
- Provide at least one visual prompt (points, bbox, or mask)
- **Output always matches input frame count**: If you input 100 frames, you get 100 masks
- Frames before/after the tracked range will have empty (blank) masks
  - Example: 100 frames, annotation_frame_idx=50, direction=forward → frames 0-49 are blank, 50-99 are tracked
  - Example: 100 frames, annotation_frame_idx=50, direction=backward → frames 0-50 are tracked, 51-99 are blank
  - Example: 100 frames, annotation_frame_idx=50, direction=bidirectional → all frames 0-99 are tracked

**Input Combination Behavior:**

You can combine different input types for powerful segmentation control:

| Input Combination | Behavior |
|-------------------|----------|
| **Points only** | Standard point-based segmentation |
| **Bbox only** | Segment object within bounding box |
| **Mask only** | Track the masked region |
| **Mask + Positive points** | Only positive points **inside the mask** are used to refine which part of the masked region to segment |
| **Mask + Negative points** | All negative points are used to exclude regions from the mask |
| **Mask + Positive + Negative** | Positive points inside mask refine the region, negative points exclude areas |

**Example Use Cases:**
- **Rough mask + precise points**: Draw a rough mask around a person, then add positive points on their face to focus the segmentation
- **Mask + negative exclusions**: Mask an object, add negative points on unwanted parts (e.g., exclude a hand from a person mask)
- **Point filtering**: Positive points outside the mask boundary are automatically ignored, preventing accidental selections

**⚠ Important Note on Negative Points with Masks:**
- Negative points work best when placed **inside or near** the masked region
- Negative points far outside the mask (>50 pixels away) may cause unexpected results or empty segmentation
- You'll receive a warning in the console if negative points are too far from the mask

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
   └─→ model

2. Load Video Frames
   └─→ frames

3. SeC Video Segmentation
   ├─ model: from (1)
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
- **CUDA GPU**: Recommended (CPU supported but significantly slower)
- **VRAM**: See GPU VRAM recommendations below
  - Can reduce significantly by enabling `offload_video_to_cpu` (~3% speed penalty)

**Note on CPU Mode:**
- CPU inference automatically uses float32 precision (bfloat16/float16 not supported on CPU)
- Expect significantly slower performance compared to GPU (~10-20x slower depending on hardware)
- Not recommended for production use, mainly for testing or systems without GPUs

## GPU VRAM Recommendations

Based on extensive testing, here are recommended configurations for different GPU VRAM tiers:

### 8-10GB VRAM
**Ideal for:** Short clips, lower resolutions
- **Resolution**: 256x256 to 512x384
- **Frame Count**: Up to 50 frames
- **Settings**:
  - `offload_video_to_cpu: True` (saves 2-3GB VRAM)
  - `torch_dtype: bfloat16`
  - `mllm_memory_size: 5-10` (optional further optimization)
- **Expected Performance**: 6-10 it/s

### 12-14GB VRAM
**Ideal for:** Standard videos, medium duration
- **Resolution**: 512x384 to 720p
- **Frame Count**: 100-200 frames
- **Settings**:
  - `offload_video_to_cpu: False` (better performance)
  - `torch_dtype: bfloat16`
  - `mllm_memory_size: 20` (default - maximum quality)
- **Expected Performance**: 5-6 it/s
- **Note**: Can handle 200 frames at 512x384 using ~13.5GB VRAM

### 16-20GB VRAM
**Ideal for:** HD videos, longer clips
- **Resolution**: 720p to 1080p
- **Frame Count**: 200-500 frames
- **Settings**:
  - `offload_video_to_cpu: False`
  - `torch_dtype: bfloat16`
  - `mllm_memory_size: 20` (maximum semantic context)
- **Expected Performance**: 4-6 it/s
- **Note**: 500 frames at 512x384 uses ~17GB VRAM

### 24GB+ VRAM
**Ideal for:** 4K video, professional workflows
- **Resolution**: 1080p to 4K
- **Frame Count**: 500+ frames
- **Settings**:
  - `offload_video_to_cpu: False`
  - `torch_dtype: bfloat16`
  - `mllm_memory_size: 20`
- **Expected Performance**: 4-5 it/s for 4K
- **Note**: 4K videos (30 frames) use ~11.5GB VRAM with plenty of headroom

### Understanding mllm_memory_size

The `mllm_memory_size` parameter controls how many historical keyframes SeC's Large Vision-Language Model uses for semantic understanding:

- **What it does**: Stores frame references for the LVLM to analyze when scene changes occur
- **VRAM impact**: None - testing shows values 3-20 use identical VRAM (~11-13GB for typical videos)
- **Compute impact**: Higher values mean more frames processed through the vision encoder on scene changes
- **Quality trade-off**: More keyframes = better object concept understanding in complex scenes
- **Recommended**: Keep at 20 (default) for best quality. Only reduce if targeting very low-end GPUs (8GB or less)

**Why doesn't it affect VRAM?** The parameter stores lightweight frame indices and mask arrays, not full frame tensors. When scene changes occur, frames are loaded from disk on-demand for LVLM processing.

## Attribution

This node implements the **SeC-4B** model developed by OpenIXCLab.

- **Model Repository**: [OpenIXCLab/SeC-4B](https://huggingface.co/OpenIXCLab/SeC-4B)
- **Paper**: [arXiv:2507.15852](https://arxiv.org/abs/2507.15852)
- **Official Implementation**: [github.com/OpenIXCLab/SeC](https://github.com/OpenIXCLab/SeC)
- **License**: Apache 2.0

**Dataset**: The original work includes the [SeCVOS Benchmark](https://huggingface.co/datasets/OpenIXCLab/SeCVOS) dataset.

## Troubleshooting

**Model download issues**:
- Ensure you have ~8.5GB disk space and internet connection
- Model auto-downloads to `ComfyUI/models/sams/SeC-4B/` on first use
- Check console for download progress and any error messages

**CUDA out of memory**:
- Enable `offload_video_to_cpu` (saves 2-3GB VRAM, only ~3% slower)
- Try `float16` precision instead of `bfloat16`
- Process fewer frames at once (split video into smaller batches)
- See GPU VRAM recommendations above for your hardware tier

**Slow inference**:
- Enable `use_flash_attn` in model loader (requires Flash Attention 2)
- Disable `offload_video_to_cpu` if you have sufficient VRAM
- Use `bfloat16` precision (default)

**Empty masks**: Provide clearer visual prompts or try different frame

---

*Self-contained ComfyUI nodes - just install and segment!* 🎉