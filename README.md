# ComfyUI SeC Nodes

**Self-contained** ComfyUI custom nodes for **SeC (Segment Concept)** - a concept-driven video object segmentation framework using Large Vision-Language Models for automatic visual concept extraction.

## Features

- 🔥 **SeC Model Loader**: Load SeC models with simple, intuitive settings
- 🔥 **SeC Video Segmentation**: Advanced video object segmentation with visual prompts
- 🚀 **Self-Contained**: All SeC inference code bundled - no separate installation needed
- 🎯 **Visual Prompts**: Points, bounding boxes, and masks
- ⚡ **Bidirectional Tracking**: Track objects forward, backward, or both directions from any frame
- 🧠 **Concept-Driven**: Automatically understands object concepts using LVLMs for robust tracking

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

## Python Compatibility
- ✅ **Python 3.10-3.12** supported
- ✅ **Works with existing ComfyUI environments**  
- ✅ **No separate conda environment needed**

## Nodes

### SeC Model Loader

Loads SeC models and tokenizers for video segmentation.

**Required Inputs:**
- `model_path`: Path to SeC model (default: `models/sams/SeC-4B`)
- `torch_dtype`: Data precision - `bfloat16` (recommended), `float16`, or `float32`
- `device`: Target device - `auto` (recommended), `cuda`, or `cpu`

**Optional Inputs:**
- `use_flash_attn`: Enable Flash Attention 2 for ~2-3x faster inference (default: True)
- `allow_mask_overlap`: Allow objects to overlap naturally (default: True). Disable for strictly separate objects.

**Outputs:**
- `model`: SeC model ready for inference
- `tokenizer`: SeC tokenizer

### SeC Video Segmentation

Concept-driven video object segmentation using automatic visual concept extraction.

**Required Inputs:**
- `model`: SeC model from model loader
- `tokenizer`: SeC tokenizer from model loader
- `frames`: Sequential video frames as IMAGE batch

**Visual Prompts (provide at least one):**
- `positive_points`: Positive clicks as JSON: `'[{"x": 100, "y": 200}]'`
- `negative_points`: Negative clicks as JSON: `'[{"x": 50, "y": 50}]'`
- `bbox`: Bounding box as `"x_min,y_min,x_max,y_max"`
- `input_mask`: Binary mask from other segmentation nodes

**Optional Settings:**
- `tracking_direction`: `forward`, `backward`, or `bidirectional` (default: forward)
- `annotation_frame_idx`: Frame where prompt is applied (default: 0) - Advanced
- `object_id`: Unique ID for multi-object tracking (default: 1) - Advanced
- `max_frames_to_track`: Max frames to process, -1 for all (default: -1) - Advanced
- `mllm_memory_size`: Frames in multimodal memory (default: 7) - Advanced

**Outputs:**
- `masks`: Segmentation masks for each tracked frame
- `object_ids`: Object IDs corresponding to masks

## Tracking Directions

### 🔄 **Bidirectional Tracking** (Recommended)
- **Use case**: Object clearest in middle of video
- **How**: Annotate clear middle frame, tracks both directions automatically
- **Example**: Frame 50 annotation → tracks frames 0-99

### ➡️ **Forward Tracking** 
- **Use case**: Standard chronological tracking
- **How**: From annotation frame to end of video

### ⬅️ **Backward Tracking**
- **Use case**: Reverse temporal analysis  
- **How**: From annotation frame to beginning of video

## Usage Examples

### Basic Point Segmentation
1. Load video frames
2. Use **SeC Model Loader** with default settings
3. **SeC Video Segmentation**:
   - Positive points: `'[{"x": 200, "y": 300}]'`
   - Annotation frame: 0
   - Tracking direction: forward

### Middle Frame Bidirectional
Perfect for clearest object view in middle:
1. Load video frames
2. **SeC Video Segmentation**:
   - Positive points: `'[{"x": 300, "y": 200}]'`
   - **Annotation frame: 50** (middle frame)
   - **Tracking direction: bidirectional**
   - Tracks entire video from clear middle frame

### Bounding Box Segmentation
1. **SeC Video Segmentation**:
   - Bbox: `"100,50,400,350"`
   - SeC automatically understands the object concept

### Multiple Points
Click multiple points on the same object:
1. **SeC Video Segmentation**:
   - Positive points: `'[{"x": 100, "y": 200}, {"x": 150, "y": 250}]'`
   - More points = more robust tracking

## Technical Notes

- **Self-contained**: All SeC inference code bundled in node
- **No git clone needed**: Everything included in node folder
- **Automatic optimization**: Adapts computation based on scene complexity
- **Memory management**: Configurable memory size for long videos
- **Multiple architectures**: Supports various LLM backbones internally

## Links
- **SeC Paper**: [arXiv:2507.15852](https://arxiv.org/abs/2507.15852)
- **SeC Model**: [🤗 HuggingFace](https://huggingface.co/OpenIXCLab/SeC-4B)
- **SeC Dataset**: [🤗 SeCVOS Dataset](https://huggingface.co/datasets/OpenIXCLab/SeCVOS)
- **Original Repository**: [GitHub - SeC](https://github.com/OpenIXCLab/SeC)

## Requirements
- PyTorch (likely already with ComfyUI)
- Python packages listed in requirements.txt
- CUDA GPU recommended for performance

The nodes are **completely self-contained** - just install dependencies and start segmenting! 🎉