# ComfyUI SeC Nodes

ComfyUI custom nodes for SeC (Segment Concept) - a concept-driven video object segmentation framework.

## Features

- **SeC Model Loader**: Load SeC models with configurable settings
- **SeC Video Segmentation**: Perform video object segmentation with multiple prompt types

## Installation

1. Copy the `comfyui_sec_nodes` folder to your ComfyUI `custom_nodes` directory
2. Install the SeC dependencies by following the main project setup instructions
3. Install additional node dependencies: `pip install -r comfyui_sec_nodes/requirements.txt`
4. Restart ComfyUI

### Windows Installation Notes

**PyTorch with CUDA**: For best performance on Windows, install PyTorch with CUDA support using the official command from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 12.1 (check pytorch.org for latest)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Alternative OpenCV**: If you encounter OpenCV issues, the requirements.txt uses `opencv-python-headless` which is more Windows-compatible than the full `opencv-python` package.

## Nodes

### SeC Model Loader

Loads SeC models and tokenizers for video segmentation tasks.

**Inputs:**
- `model_path`: Path to SeC model or HuggingFace model ID (default: "OpenIXCLab/SeC-4B")
- `torch_dtype`: Data type for inference (bfloat16/float16/float32)
- `use_flash_attn`: Enable flash attention for faster inference
- `device`: Device to load model on (auto/cuda/cpu)
- `hydra_overrides`: Configuration overrides
- `grounding_maskmem_num`: Memory frames for grounding encoder

**Outputs:**
- `model`: Loaded SeC model
- `tokenizer`: Loaded tokenizer

### SeC Video Segmentation

Performs video object segmentation using various prompt types.

**Required Inputs:**
- `model`: SeC model from model loader
- `tokenizer`: SeC tokenizer from model loader  
- `frames`: Sequential video frames as IMAGE batch
- `annotation_frame_idx`: Frame index to annotate (0-based)
- `object_id`: Unique object ID for tracking

**Optional Inputs:**
- `positive_points`: Positive click points as "x1,y1;x2,y2" format
- `negative_points`: Negative click points as "x1,y1;x2,y2" format
- `bbox`: Bounding box as "x_min,y_min,x_max,y_max"
- `input_mask`: Input mask for initialization
- `text_prompt`: Text description of object to segment
- `start_frame_idx`: Start frame for propagation
- `max_frames_to_track`: Maximum frames to track (-1 for all)
- `tracking_direction`: Direction of temporal tracking (forward/backward/bidirectional)
- `mllm_memory_size`: Memory size for multimodal reasoning
- `output_stride`: Output every N-th frame

**Outputs:**
- `masks`: Segmentation masks for each frame
- `object_ids`: Object IDs corresponding to each mask

## Temporal Tracking Directions

SeC supports flexible temporal tracking from any annotation frame:

### Forward Tracking (Default)
- **tracking_direction**: "forward"  
- Tracks from annotation frame → end of video
- Best for: First frame annotations, typical video analysis

### Backward Tracking  
- **tracking_direction**: "backward"
- Tracks from annotation frame → beginning of video
- Best for: Last frame annotations, reverse chronological analysis

### Bidirectional Tracking ✨
- **tracking_direction**: "bidirectional" 
- Automatically tracks in BOTH directions from annotation frame
- Implementation: Performs forward tracking, resets state, then backward tracking
- Best for: **Middle frame annotations** - annotate a clear frame and track throughout entire video
- **Use case**: When the clearest view of your object is in the middle of the video

## Usage Examples

### Basic Point-based Segmentation

1. Load video frames using standard ComfyUI image nodes
2. Use **SeC Model Loader** to load the model
3. Connect to **SeC Video Segmentation** with:
   - Positive points: "200,300;250,100" (two positive clicks)
   - Annotation frame: 0
   - Tracking direction: "forward"
   - Object ID: 1

### Middle Frame Bidirectional Tracking 🎯

Perfect for when the object is clearest in the middle of the video:

1. Set up model and frames as above
2. Use **SeC Video Segmentation** with:
   - Positive points: "300,200" (click on clear object view)
   - **Annotation frame: 50** (middle frame where object is clearly visible)
   - **Tracking direction: "bidirectional"** 
   - Object ID: 1

This will track the object from frame 50 forward to the end, AND backward to the beginning.

### Bounding Box Segmentation

1. Set up model and frames as above
2. Use **SeC Video Segmentation** with:
   - Bbox: "100,50,400,350" (x_min,y_min,x_max,y_max)
   - Annotation frame: 0
   - Tracking direction: "forward"

### Text-guided Segmentation

1. Set up model and frames as above  
2. Use **SeC Video Segmentation** with:
   - Text prompt: "person wearing red shirt"
   - Annotation frame: 0
   - Tracking direction: "forward"

### Reverse Timeline Analysis

For analyzing how an object appeared in the past:

1. Set up model and frames as above
2. Use **SeC Video Segmentation** with:
   - Bbox: "200,100,500,400" (final position of object)
   - **Annotation frame: 99** (last frame)
   - **Tracking direction: "backward"**
   - Object ID: 1

## Technical Implementation Notes

### Bidirectional Tracking Implementation
The "bidirectional" option is a ComfyUI enhancement that automatically:
1. Performs forward tracking from annotation frame to end
2. Resets the inference state 
3. Re-adds the initial annotation
4. Performs backward tracking from annotation frame to beginning
5. Combines results from both directions

This provides seamless bidirectional tracking while working within SeC's unidirectional-per-call architecture.

### General Notes
- At least one prompt type (points, bbox, mask, or text) must be provided
- The model supports dynamic scene understanding and adapts computational effort based on complexity
- **Best practice**: Choose annotation frames where your target object is clearly visible and unoccluded
- The node temporarily saves frames to disk during processing and cleans up automatically
- For challenging sequences, consider using multiple annotation frames with different object IDs

## Requirements

- PyTorch with CUDA support
- transformers
- PIL/Pillow
- numpy
- opencv-python
- All SeC project dependencies

## Model Support

- Supports official SeC models from OpenIXCLab
- Compatible with various backbone architectures (Llama, InternLM2, Phi3, Qwen2)  
- Configurable LoRA usage and memory settings