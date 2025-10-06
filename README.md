# ComfyUI SeC Nodes

**Self-contained** ComfyUI custom nodes for **SeC (Segment Concept)** - a concept-driven video object segmentation framework that intelligently combines visual features with semantic reasoning.

## Features

- 🔥 **SeC Model Loader**: Load SeC models with configurable settings
- 🔥 **SeC Video Segmentation**: Advanced video object segmentation with multiple prompt types
- 🚀 **Self-Contained**: All SeC inference code bundled - no separate installation needed
- 🎯 **Multiple Prompt Types**: Points, bounding boxes, masks, and text descriptions
- ⚡ **Bidirectional Tracking**: Track objects forward, backward, or both directions from any frame
- 🧠 **Concept-Driven**: Adapts computational effort based on scene complexity

## Quick Start Installation

### 1. Copy Node Folder
```bash
# Copy the entire comfyui_sec_nodes folder to your ComfyUI custom_nodes directory
cp -r comfyui_sec_nodes /path/to/ComfyUI/custom_nodes/
```

### 2. Install Dependencies
```bash
# Install required Python packages
pip install -r comfyui_sec_nodes/requirements.txt
```

### 3. Download SeC Model
Download the SeC model from [🤗HuggingFace](https://huggingface.co/OpenIXCLab/SeC-4B):
```bash
# Using huggingface_hub (recommended)
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenIXCLab/SeC-4B', local_dir='./SeC-4B')"

# Or using git lfs
git lfs clone https://huggingface.co/OpenIXCLab/SeC-4B
```

### 4. Restart ComfyUI
That's it! The nodes will appear in the "SeC" category.

## Python Compatibility
- ✅ **Python 3.10-3.12** supported
- ✅ **Works with existing ComfyUI environments**  
- ✅ **No separate conda environment needed**

## Nodes

### SeC Model Loader

Loads SeC models and tokenizers for video segmentation.

**Inputs:**
- `model_path`: HuggingFace model ID or local path (default: "OpenIXCLab/SeC-4B")
- `torch_dtype`: Data precision (bfloat16/float16/float32)
- `use_flash_attn`: Enable Flash Attention 2 for speed
- `device`: Target device (auto/cuda/cpu)  
- `hydra_overrides`: Configuration overrides
- `grounding_maskmem_num`: Memory frames for temporal consistency

**Outputs:**
- `model`: SeC model ready for inference
- `tokenizer`: SeC tokenizer

### SeC Video Segmentation

Advanced video object segmentation with concept-driven understanding.

**Required Inputs:**
- `model`: SeC model from model loader
- `tokenizer`: SeC tokenizer from model loader
- `frames`: Sequential video frames as IMAGE batch
- `annotation_frame_idx`: Frame to annotate (0-based)
- `object_id`: Unique object identifier

**Prompt Inputs (choose one or more):**
- `positive_points`: Positive clicks as "x1,y1;x2,y2"
- `negative_points`: Negative clicks as "x1,y1;x2,y2"
- `bbox`: Bounding box as "x_min,y_min,x_max,y_max"
- `input_mask`: Binary mask image
- `text_prompt`: Natural language description

**Tracking Options:**
- `tracking_direction`: "forward", "backward", or "bidirectional"
- `start_frame_idx`: Start frame for propagation
- `max_frames_to_track`: Frame limit (-1 for all)
- `mllm_memory_size`: Frames in multimodal memory
- `output_stride`: Output every N-th frame

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
   - Positive points: "200,300"
   - Annotation frame: 0
   - Tracking direction: "forward"

### Middle Frame Bidirectional
Perfect for clearest object view in middle:
1. Load video frames  
2. **SeC Video Segmentation**:
   - Positive points: "300,200"
   - **Annotation frame: 50** (middle frame)
   - **Tracking direction: "bidirectional"**
   - Tracks entire video from clear middle frame

### Text-Guided Segmentation
1. **SeC Video Segmentation**:
   - Text prompt: "person wearing red shirt"
   - Annotation frame: 0
   - Leverages concept understanding

### Bounding Box + Text
1. **SeC Video Segmentation**:
   - Bbox: "100,50,400,350"
   - Text prompt: "main character"
   - Combines spatial and semantic cues

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