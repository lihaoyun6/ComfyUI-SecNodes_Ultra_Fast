# ComfyUI SeC Nodes

**Self-contained** ComfyUI custom nodes for **SeC (Segment Concept)** - State-of-the-art video object segmentation that outperforms SAM 2.1, utilizing the SeC-4B model developed by OpenIXCLab.

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

## Demo

https://github.com/user-attachments/assets/5cc6677e-4a9d-4e55-801d-b92305a37725

*Example: SeC tracking an object through scene changes and dynamic movement*



https://github.com/user-attachments/assets/9e99d55c-ba8e-4041-985e-b95cbd6dd066

*Example: SAM fails to track correct dog for some scenes*

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

**If using ComfyUI Portable:**
```bash
cd *install_location*/ComfyUI/custom_nodes/Comfyui-SecNodes
*install_location*\python_embeded\python.exe -m pip install -r requirements.txt
```

**If using standard Python installation:**
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
| *mllm_memory_size* | INT | 12 | Number of keyframes for semantic understanding (affects compute on scene changes, not VRAM). Original paper used 7. |
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

- **Python**: 3.10-3.12 (3.12 recommended)
  - Python 3.13: Not recommended - experimental support with known dependency installation issues
- **PyTorch**: 2.6.0+ (included with ComfyUI)
- **CUDA**: 11.8+ for GPU acceleration
- **CUDA GPU**: Recommended (CPU supported but significantly slower)
- **VRAM**: See GPU VRAM recommendations below
  - Can reduce significantly by enabling `offload_video_to_cpu` (~3% speed penalty)

**Note on CPU Mode:**
- CPU inference automatically uses float32 precision (bfloat16/float16 not supported on CPU)
- Expect significantly slower performance compared to GPU (~10-20x slower depending on hardware)
- Not recommended for production use, mainly for testing or systems without GPUs

**Flash Attention 2 (Optional):**
- Provides ~2x speedup but requires specific hardware
- **GPU Requirements**: Ampere/Ada/Hopper architecture only (RTX 30/40 series, A100, H100)
  - Does NOT work on RTX 20 series (Turing) or older GPUs
- **CUDA**: 12.0+ required
- **Windows + Python 3.12**: Use pre-compiled wheels or disable flash attention
- The node automatically falls back to standard attention if Flash Attention is unavailable

## GPU VRAM Recommendations

| VRAM | Resolution | Frames | Key Settings | Performance |
|------|-----------|--------|--------------|-------------|
| **8-10GB** | 256x256 - 512x384 | Up to 50 | `offload_video_to_cpu: True`<br>`mllm_memory_size: 5-10` | 6-10 it/s |
| **12-14GB** | 512x384 - 720p | 100-200 | Default settings work well<br>Can handle 200 frames @ 512x384 (~13.5GB) | 5-6 it/s |
| **16-20GB** | 720p - 1080p | 200-500 | `mllm_memory_size: 12-15`<br>500 frames @ 512x384 uses ~17GB | 4-6 it/s |
| **24GB+** | 1080p - 4K | 500+ | `mllm_memory_size: 15-20` for max quality<br>4K (30 frames) uses ~11.5GB | 4-5 it/s |

**Quick Tips:**
- Low on VRAM? Enable `offload_video_to_cpu` (saves 2-3GB, only ~3% slower)
- Use `torch_dtype: bfloat16` for best balance of speed and quality
- Lower `mllm_memory_size` (5-10) if you need to squeeze into limited VRAM
### Understanding mllm_memory_size

The `mllm_memory_size` parameter controls how many historical keyframes SeC's Large Vision-Language Model uses for semantic understanding:

- **What it does**: Stores frame references (first frame + last N-1 frames) for the LVLM to analyze when scene changes occur
- **VRAM impact**: None - testing shows values 3-20 use similar VRAM (~11-13GB for typical videos)
- **Compute impact**: Higher values mean more frames processed through the vision encoder on scene changes
- **Quality trade-off**: More keyframes = better object concept understanding in complex scenes, but diminishing returns after ~10-12 frames
- **Original research**: SeC paper used 7 and achieved SOTA performance (+11.8 over SAM 2.1), emphasizing "quality over quantity" of keyframes

**Recommended Values:**
- **Default (12)**: Balanced approach - higher than paper's 7 for extra context, but not excessive
- **Low (5-7)**: Faster inference on simple videos, matches original research setup
- **High (15-20)**: Maximum semantic context for very complex videos (no VRAM penalty)

**Why doesn't it affect VRAM?** The parameter stores lightweight frame indices and mask arrays, not full frame tensors. When scene changes occur, frames are loaded from disk on-demand for LVLM processing. The underlying SAM2 architecture supports up to 22 frames.

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
