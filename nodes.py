# Uses SeC-4B model from OpenIXCLab
# Model: https://huggingface.co/OpenIXCLab/SeC-4B
# Licensed under Apache 2.0

import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import sys

from .inference.configuration_sec import SeCConfig
from .inference.modeling_sec import SeCModel
from transformers import AutoTokenizer


def find_sec_model():
    """
    Find SeC-4B model in registered 'sams' folder paths.
    Returns the path to the model directory if found, None otherwise.
    """
    try:
        sams_paths = folder_paths.get_folder_paths("sams")
    except KeyError:
        # 'sams' folder type not registered yet
        return None

    for sams_dir in sams_paths:
        model_path = os.path.join(sams_dir, "SeC-4B")
        if os.path.exists(model_path) and os.path.isdir(model_path):
            # Verify required files exist
            config_exists = os.path.exists(os.path.join(model_path, "config.json"))
            model_exists = (
                os.path.exists(os.path.join(model_path, "model.safetensors")) or
                os.path.exists(os.path.join(model_path, "model.safetensors.index.json")) or  # Sharded safetensors
                os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(model_path, "pytorch_model.bin.index.json"))  # Sharded bin
            )
            tokenizer_exists = os.path.exists(os.path.join(model_path, "tokenizer_config.json"))

            if config_exists and model_exists and tokenizer_exists:
                return model_path

    return None


def download_sec_model():
    """
    Download SeC-4B model from HuggingFace to the first registered 'sams' folder.
    Returns the path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for model download. "
            "Please install it: pip install huggingface_hub>=0.20.0"
        ) from e

    try:
        sams_paths = folder_paths.get_folder_paths("sams")
    except Exception as e:
        raise RuntimeError(f"Could not access model folder paths: {e}") from e

    if not sams_paths:
        raise RuntimeError("No 'sams' folder paths registered. Please check your ComfyUI installation.")

    destination = os.path.join(sams_paths[0], "SeC-4B")

    print("=" * 80)
    print("SeC-4B model not found. Downloading from HuggingFace...")
    print(f"Repository: OpenIXCLab/SeC-4B")
    print(f"Destination: {destination}")
    print(f"Size: ~8.5 GB - This may take several minutes...")
    print("=" * 80)

    # Create directory if it doesn't exist
    try:
        os.makedirs(destination, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise RuntimeError(
            f"Cannot create model directory at {destination}. "
            f"Please check permissions. Error: {e}"
        ) from e

    # Check disk space (rough estimate)
    try:
        import shutil
        stat = shutil.disk_usage(os.path.dirname(destination))
        free_gb = stat.free / (1024**3)
        if free_gb < 10:
            print(f"⚠ Warning: Low disk space ({free_gb:.1f} GB free). Download requires ~8.5 GB.")
    except Exception:
        pass  # Not critical

    try:
        snapshot_download(
            repo_id="OpenIXCLab/SeC-4B",
            local_dir=destination,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            raise RuntimeError(
                f"Network error while downloading model: {e}\n"
                "Please check your internet connection and try again."
            ) from e
        elif "space" in error_msg or "disk" in error_msg:
            raise RuntimeError(
                f"Insufficient disk space: {e}\n"
                "Model download requires ~8.5 GB free space."
            ) from e
        else:
            raise RuntimeError(f"Failed to download model from HuggingFace: {e}") from e

    print("=" * 80)
    print(f"✓ SeC-4B model downloaded successfully!")
    print(f"✓ Location: {destination}")
    print("=" * 80)

    return destination


class SeCModelLoader:
    """
    ComfyUI node for loading SeC (Segment Concept) models
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically build device list based on available GPUs
        device_choices = ["auto", "cpu"]

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                device_choices.append(f"gpu{i}")

        return {
            "required": {
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Data precision for model inference. bfloat16 recommended for best speed/quality balance. CPU mode automatically uses float32."
                }),
                "device": (device_choices, {
                    "default": "auto",
                    "tooltip": "Device: auto (gpu0 if available, else CPU), cpu, gpu0/gpu1/etc (specific GPU)"
                })
            },
            "optional": {
                "use_flash_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster inference. Automatically disabled for float32 precision (requires float16/bfloat16)."
                }),
                "allow_mask_overlap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow tracked objects to overlap. Disable for strictly separate objects."
                })
            }
        }
    
    RETURN_TYPES = ("SEC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SeC"
    TITLE = "SeC Model Loader"
    
    def load_model(self, torch_dtype, device, use_flash_attn=True, allow_mask_overlap=True):
        """Load SeC model"""

        # Find or download the SeC-4B model
        model_path = find_sec_model()

        if model_path is None:
            # Model not found, download it
            try:
                model_path = download_sec_model()
            except Exception as e:
                raise RuntimeError(f"Failed to download SeC-4B model: {str(e)}")
        else:
            print("=" * 80)
            print(f"✓ Found SeC-4B model at: {model_path}")
            print("=" * 80)

        # Handle device selection
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            # Map gpu0 -> cuda:0, gpu1 -> cuda:1, etc.
            try:
                gpu_num = int(device[3:])  # Extract number after "gpu"
                if torch.cuda.is_available():
                    available_gpus = torch.cuda.device_count()
                    if gpu_num >= available_gpus:
                        raise ValueError(f"GPU {gpu_num} not available. System has {available_gpus} GPU(s) (0-{available_gpus-1})")
                else:
                    raise ValueError(f"CUDA not available but GPU device '{device}' was selected")
                device = f"cuda:{gpu_num}"
            except (ValueError, IndexError) as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid GPU device format: '{device}'. Expected format: 'gpu0', 'gpu1', etc.")
                raise

        # Force float32 for CPU mode to avoid dtype mismatches
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[torch_dtype]

        if device == "cpu" and torch_dtype != torch.float32:
            print(f"⚠ CPU mode requires float32 precision. Overriding {torch_dtype} -> float32")
            torch_dtype = torch.float32

        # Flash Attention requires float16/bfloat16 - auto-disable for float32
        if torch_dtype == torch.float32 and use_flash_attn:
            print("⚠ Flash Attention requires float16/bfloat16 precision. Disabling Flash Attention for float32.")
            print("  Note: Inference will use standard attention (slower but compatible with float32)")
            use_flash_attn = False

        hydra_overrides_extra = []
        overlap_value = "false" if allow_mask_overlap else "true"
        hydra_overrides_extra.append(f"++model.non_overlap_masks={overlap_value}")

        try:
            config = SeCConfig.from_pretrained(model_path)
            config.hydra_overrides_extra = hydra_overrides_extra

            load_kwargs = {
                "config": config,
                "torch_dtype": torch_dtype,
                "use_flash_attn": use_flash_attn,
            }

            # Configure device_map based on device selection
            if device.startswith("cuda:"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                load_kwargs["device_map"] = {"": device}  # Force to specific GPU
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"Loading SeC model on {device}...")
            else:
                # CPU mode
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"Loading SeC model on CPU (float32)...")

            model = SeCModel.from_pretrained(model_path, **load_kwargs).eval()

            # Convert model to target device and dtype
            if device.startswith("cuda") and torch_dtype != torch.float32:
                print(f"Converting model to {torch_dtype}...")
                model = model.to(dtype=torch_dtype)
            else:
                # CPU mode - ensure everything is float32
                model = model.to(device=device, dtype=torch_dtype)

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

            if device.startswith("cuda") and torch_dtype != torch.float32:
                print(f"Installing dtype conversion hooks...")

                def dtype_conversion_hook(module, args, kwargs):
                    try:
                        module_dtype = None
                        for param in module.parameters():
                            module_dtype = param.dtype
                            break

                        if module_dtype is None:
                            return args, kwargs

                        if isinstance(module, torch.nn.Embedding):
                            return args, kwargs

                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                if arg.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_args.append(arg)
                                elif arg.dtype != module_dtype:
                                    new_args.append(arg.to(dtype=module_dtype))
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)

                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_kwargs[k] = v
                                elif v.dtype != module_dtype:
                                    new_kwargs[k] = v.to(dtype=module_dtype)
                                else:
                                    new_kwargs[k] = v
                            else:
                                new_kwargs[k] = v

                        return tuple(new_args), new_kwargs
                    except Exception:
                        return args, kwargs

                for module in model.modules():
                    if len(list(module.parameters(recurse=False))) > 0:
                        module.register_forward_pre_hook(dtype_conversion_hook, with_kwargs=True)

            print(f"SeC model loaded successfully on {device}")

            return (model,)

        except Exception as e:
            raise RuntimeError(f"Failed to load SeC model: {str(e)}")


class SeCVideoSegmentation:
    """
    SeC Video Object Segmentation - Concept-driven video segmentation using multimodal understanding.
    
    Performs intelligent video object segmentation by combining visual features with semantic reasoning.
    Supports multiple prompt types and adapts computational effort based on scene complexity.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEC_MODEL", {
                    "tooltip": "SeC model loaded from SeCModelLoader node"
                }),
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                })
            },
            "optional": {
                "positive_points": ("STRING", {
                    "default": "",
                    "tooltip": "Positive click coordinates as JSON: '[{\"x\": 63, \"y\": 782}]'"
                }),
                "negative_points": ("STRING", {
                    "default": "",
                    "tooltip": "Negative click coordinates as JSON: '[{\"x\": 100, \"y\": 200}]'"
                }),
                "bbox": ("STRING", {
                    "default": "",
                    "tooltip": "Bounding box as 'x_min,y_min,x_max,y_max'"
                }),
                "input_mask": ("MASK", {
                    "tooltip": "Binary mask for object initialization"
                }),
                "tracking_direction": (["forward", "backward", "bidirectional"], {
                    "default": "forward",
                    "tooltip": "Tracking direction from annotation frame"
                }),
                "annotation_frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Advanced: Frame where initial prompt is applied"
                }),
                "object_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Advanced: Unique ID for multi-object tracking"
                }),
                "max_frames_to_track": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Advanced: Max frames to process (-1 for all)"
                }),
                "mllm_memory_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Advanced: Number of keyframes stored for semantic understanding (does not affect VRAM, only compute on scene changes)."
                }),
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Memory: Offload video frames to CPU (saves significant GPU memory, ~3% slower)"
                })
            }
        }
    
    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("masks", "object_ids") 
    FUNCTION = "segment_video"
    CATEGORY = "SeC"
    TITLE = "SeC Video Segmentation"
    DESCRIPTION = ("Concept-driven video object segmentation using Large Vision-Language Models for visual concept extraction. "
                   "Provide visual prompts (points/bbox/mask) and SeC automatically understands the object concept for robust tracking.")
    
    def parse_points(self, points_str, image_shape=None):
        """Parse point coordinates from JSON string and validate bounds.

        Returns:
            tuple: (points_array, labels_array, validation_errors) where validation_errors
                   is a list of error messages, or (None, None, errors) if all points invalid
        """
        import json

        if not points_str or not points_str.strip():
            return None, None, []

        try:
            points_list = json.loads(points_str)

            if not isinstance(points_list, list):
                raise ValueError(f"Points must be a JSON array, got {type(points_list).__name__}")

            if len(points_list) == 0:
                return None, None, []

            points = []
            validation_errors = []

            for i, point_dict in enumerate(points_list):
                if not isinstance(point_dict, dict):
                    err = f"Point {i} is not a dictionary"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

                if 'x' not in point_dict or 'y' not in point_dict:
                    err = f"Point {i} missing 'x' or 'y' key"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

                try:
                    x = float(point_dict['x'])
                    y = float(point_dict['y'])

                    # Validate coordinates are non-negative
                    if x < 0 or y < 0:
                        err = f"Point {i} has negative coordinates ({x}, {y})"
                        print(f"Warning: {err}, skipping")
                        validation_errors.append(err)
                        continue

                    # Validate within image bounds if provided
                    if image_shape is not None:
                        height, width = image_shape[1], image_shape[2]  # [batch, height, width, channels]
                        if x >= width or y >= height:
                            err = f"Point {i} ({x}, {y}) outside image bounds ({width}x{height})"
                            print(f"Warning: {err}, skipping")
                            validation_errors.append(err)
                            continue

                    points.append([x, y])

                except (ValueError, TypeError) as e:
                    err = f"Could not convert point {i} coordinates to float: {e}"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

            if not points:
                return None, None, validation_errors

            return np.array(points, dtype=np.float32), np.ones(len(points), dtype=np.int32), validation_errors

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in points: {str(e)}")
        except Exception as e:
            print(f"Error parsing points: {e}")
            return None, None, [str(e)]
    
    def parse_bbox(self, bbox_str):
        """Parse bounding box from string and validate"""
        if not bbox_str or not bbox_str.strip():
            return None

        try:
            coords = [float(x.strip()) for x in bbox_str.strip().split(',')]

            if len(coords) != 4:
                raise ValueError(f"Bounding box must have 4 coordinates, got {len(coords)}")

            x1, y1, x2, y2 = coords

            # Validate coordinates are sensible
            if x1 >= x2:
                raise ValueError(f"Invalid bbox: x1 ({x1}) must be < x2 ({x2})")
            if y1 >= y2:
                raise ValueError(f"Invalid bbox: y1 ({y1}) must be < y2 ({y2})")

            if x1 < 0 or y1 < 0:
                raise ValueError(f"Bounding box coordinates must be non-negative, got x1={x1}, y1={y1}")

            return np.array(coords, dtype=np.float32)

        except ValueError as e:
            if "could not convert" in str(e):
                raise ValueError(f"Invalid bbox format: '{bbox_str}'. Expected format: 'x1,y1,x2,y2' with numeric values")
            raise  # Re-raise our custom errors
    
    def tensor_to_pil_images(self, tensor):
        """Convert tensor to list of PIL images"""
        images = []
        for i in range(tensor.shape[0]):
            img_array = (tensor[i] * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img_array, mode='RGB')
            images.append(pil_img)
        return images

    def pil_images_to_tensor(self, pil_images):
        """Convert list of PIL images to tensor"""
        if not pil_images:
            return torch.empty(0)

        arrays = []
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            arrays.append(arr)

        tensor = torch.from_numpy(np.stack(arrays))
        return tensor

    def mask_to_tensor(self, mask_array):
        """Convert numpy mask to ComfyUI MASK tensor (2D grayscale)"""
        if mask_array.ndim > 2:
            mask_array = mask_array[..., 0] if mask_array.shape[-1] <= 4 else mask_array[0]

        mask_tensor = torch.from_numpy(mask_array.astype(np.float32))
        return mask_tensor
    
    def save_frames_temporarily(self, pil_images, temp_dir=None):
        """Save frames temporarily for video processing"""
        import tempfile

        # Use system temp directory for cross-platform compatibility
        if temp_dir is None:
            temp_base = tempfile.gettempdir()
            temp_dir = os.path.join(temp_base, "sec_frames")

        os.makedirs(temp_dir, exist_ok=True)

        # Clean up old files safely
        try:
            for file in os.listdir(temp_dir):
                if file.endswith(('.jpg', '.png')):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except (PermissionError, OSError) as e:
                        print(f"Warning: Could not remove old frame file {file}: {e}")
        except Exception as e:
            print(f"Warning: Error during temp directory cleanup: {e}")

        frame_paths = []
        for i, img in enumerate(pil_images):
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            img.save(frame_path, 'JPEG', quality=95)
            frame_paths.append(frame_path)

        return temp_dir, frame_paths
    
    def segment_video(self, model, frames, positive_points="", negative_points="",
                     bbox="", input_mask=None, tracking_direction="forward",
                     annotation_frame_idx=0, object_id=1, max_frames_to_track=-1, mllm_memory_size=20,
                     offload_video_to_cpu=False):
        """Perform video object segmentation"""

        # === Input Validation ===
        # Validate frames tensor
        if frames is None or frames.numel() == 0:
            raise ValueError("Frames tensor is empty. Please provide at least one frame.")

        if frames.ndim != 4:
            raise ValueError(f"Frames tensor must be 4D [batch, height, width, channels], got shape {frames.shape}")

        num_frames = frames.shape[0]
        if num_frames == 0:
            raise ValueError("No frames provided. Frames tensor has 0 frames.")

        # Validate annotation_frame_idx bounds
        if annotation_frame_idx < 0:
            raise ValueError(f"annotation_frame_idx must be >= 0, got {annotation_frame_idx}")

        if annotation_frame_idx >= num_frames:
            raise ValueError(
                f"annotation_frame_idx ({annotation_frame_idx}) is out of bounds. "
                f"Video has {num_frames} frame(s), valid range is 0-{num_frames-1}"
            )

        # Validate at least one input provided
        has_input = (
            (positive_points and positive_points.strip()) or
            (negative_points and negative_points.strip()) or
            (bbox and bbox.strip()) or
            (input_mask is not None)
        )
        if not has_input:
            raise ValueError(
                "At least one visual prompt must be provided: "
                "positive_points, negative_points, bbox, or input_mask"
            )

        video_dir = None  # Track for cleanup
        try:
            pil_images = self.tensor_to_pil_images(frames)
            video_dir, frame_paths = self.save_frames_temporarily(pil_images)

            # Automatically set offload_state_to_cpu based on model device
            try:
                offload_state_to_cpu = str(model.device) == "cpu"
            except AttributeError:
                # Fallback if model doesn't have device attribute
                offload_state_to_cpu = False

            inference_state = model.grounding_encoder.init_state(
                video_path=video_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu
            )
            model.grounding_encoder.reset_state(inference_state)

            # Parse inputs with bounds checking
            pos_points, pos_labels, pos_errors = self.parse_points(positive_points, frames.shape)
            neg_points, neg_labels, neg_errors = self.parse_points(negative_points, frames.shape)
            bbox_coords = self.parse_bbox(bbox)

            # Collect validation errors for better error messages
            all_validation_errors = []
            if pos_errors:
                all_validation_errors.extend([f"Positive {err}" for err in pos_errors])
            if neg_errors:
                all_validation_errors.extend([f"Negative {err}" for err in neg_errors])

            init_mask = None

            # Step 1: Add mask if provided (establishes initial region)
            if input_mask is not None:
                # Handle both [H, W] and [B, H, W] mask formats
                if input_mask.dim() == 2:
                    mask_array = input_mask.cpu().numpy()
                elif input_mask.dim() == 3:
                    mask_array = input_mask[0].cpu().numpy()
                else:
                    raise ValueError(f"Unexpected mask dimensions: {input_mask.dim()}. Expected 2D [H,W] or 3D [B,H,W]")

                init_mask = (mask_array > 0.5).astype(np.bool_)

                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    mask=init_mask,
                )

            # Step 2: Filter positive points if mask was provided
            # Only keep positive points that fall inside the mask boundary
            if init_mask is not None and pos_points is not None:
                filtered_pos_points = []
                filtered_pos_labels = []
                for i, point in enumerate(pos_points):
                    x, y = int(point[0]), int(point[1])
                    # Check if point is within mask bounds and inside the mask
                    if 0 <= y < init_mask.shape[0] and 0 <= x < init_mask.shape[1]:
                        if init_mask[y, x]:  # Point is inside mask
                            filtered_pos_points.append(point)
                            filtered_pos_labels.append(pos_labels[i])

                # Replace pos_points with filtered version
                if filtered_pos_points:
                    pos_points = np.array(filtered_pos_points)
                    pos_labels = np.array(filtered_pos_labels, dtype=np.int32)
                else:
                    # No positive points inside mask - clear them
                    pos_points = None
                    pos_labels = None

            # Step 2b: Warn about negative points when mask is provided
            # Negative points should ideally be inside or near the mask to refine segmentation
            if init_mask is not None and neg_points is not None:
                # Find pixels in the mask
                mask_pixels = np.argwhere(init_mask)
                if len(mask_pixels) > 0:
                    points_outside = []
                    for i, point in enumerate(neg_points):
                        x, y = int(point[0]), int(point[1])
                        # Calculate minimum distance to any mask pixel
                        distances = np.sqrt(((mask_pixels[:, 0] - y) ** 2) + ((mask_pixels[:, 1] - x) ** 2))
                        min_dist = distances.min()

                        # If point is >50 pixels away from mask, warn
                        if min_dist > 50:
                            points_outside.append((i, min_dist))

                    if points_outside:
                        print(f"⚠ Warning: {len(points_outside)} negative point(s) are far from the mask region.")
                        print(f"  Negative points work best inside or near the masked object to refine segmentation.")
                        print(f"  Points far outside the mask may cause unexpected results or empty segmentation.")

            # Step 3: Combine points for refinement
            points = None
            labels = None
            if pos_points is not None and neg_points is not None:
                points = np.concatenate([pos_points, neg_points], axis=0)
                labels = np.concatenate([pos_labels, np.zeros(len(neg_points), dtype=np.int32)], axis=0)
            elif pos_points is not None:
                points = pos_points
                labels = pos_labels
            elif neg_points is not None:
                points = neg_points
                labels = np.zeros(len(neg_points), dtype=np.int32)

            # Step 4: Add points/bbox to refine the segmentation
            if points is not None or bbox_coords is not None:
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels if points is not None else None,
                    box=bbox_coords,
                )
                init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            # Ensure at least one input was provided
            if init_mask is None:
                error_msg = "At least one visual prompt (points, bbox, or mask) must be provided."
                if all_validation_errors:
                    error_msg += f" Point validation failures: {'; '.join(all_validation_errors)}"
                raise ValueError(error_msg)

            if max_frames_to_track == -1:
                max_frames_to_track = len(pil_images)

            video_segments = {}
            
            if tracking_direction == "bidirectional":
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=False,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=mllm_memory_size,
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                model.grounding_encoder.reset_state(inference_state)

                if points is not None or bbox_coords is not None:
                    _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=annotation_frame_idx,
                        obj_id=object_id,
                        points=points,
                        labels=labels if points is not None else None,
                        box=bbox_coords,
                    )
                elif input_mask is not None:
                    _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=annotation_frame_idx,
                        obj_id=object_id,
                        mask=init_mask,
                    )

                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=True,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=mllm_memory_size,
                ):
                    if out_frame_idx not in video_segments:
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
            else:
                reverse = (tracking_direction == "backward")
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=reverse,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=mllm_memory_size,
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            
            # Create output masks for all input frames
            # Frames not in video_segments get empty masks
            num_frames = len(pil_images)
            output_masks = []
            output_obj_ids = []

            for frame_idx in range(num_frames):
                if frame_idx in video_segments:
                    # Frame was tracked - use real mask
                    for obj_id, mask in video_segments[frame_idx].items():
                        mask_tensor = self.mask_to_tensor(mask)
                        output_masks.append(mask_tensor)
                        output_obj_ids.append(obj_id)
                else:
                    # Frame not tracked - use empty mask
                    empty_mask = torch.zeros(frames.shape[1], frames.shape[2])
                    output_masks.append(empty_mask)
                    output_obj_ids.append(0)

            masks_tensor = torch.stack(output_masks)
            obj_ids_tensor = torch.tensor(output_obj_ids, dtype=torch.int32)

            return (masks_tensor, obj_ids_tensor)

        except Exception as e:
            raise RuntimeError(f"SeC video segmentation failed: {str(e)}")

        finally:
            # Cleanup: Always remove temp directory and clear cache
            import shutil
            import gc

            if video_dir is not None and os.path.exists(video_dir):
                try:
                    shutil.rmtree(video_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp directory {video_dir}: {e}")

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


class CoordinatePlotter:
    """
    ComfyUI node for visualizing coordinates on images
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {
                    "default": '[{"x": 100, "y": 100}]',
                    "tooltip": "JSON coordinates to plot: '[{\"x\": 100, \"y\": 200}]'"
                })
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image to plot on. If provided, overrides width/height."
                }),
                "point_shape": (["circle", "square", "triangle"], {
                    "default": "circle",
                    "tooltip": "Shape to draw for each coordinate point"
                }),
                "point_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Size of points in pixels"
                }),
                "point_color": ("STRING", {
                    "default": "#00FF00",
                    "tooltip": "Point color as hex '#FF0000' or RGB '255,0,0'"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Canvas width (ignored if image provided)"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Canvas height (ignored if image provided)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "plot_coordinates"
    CATEGORY = "SeC"
    TITLE = "Coordinate Plotter"
    DESCRIPTION = "Visualize coordinate points on an image or blank canvas. Useful for previewing point selections."

    def parse_color(self, color_str):
        """Parse hex or RGB color string to BGR tuple for OpenCV"""
        import re

        color_str = color_str.strip()

        if color_str.startswith('#'):
            color_str = color_str[1:]

        if re.match(r'^[0-9A-Fa-f]{6}$', color_str):
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            return (b, g, r)

        if ',' in color_str:
            parts = [int(x.strip()) for x in color_str.split(',')]
            if len(parts) == 3:
                r, g, b = parts
                return (b, g, r)

        return (0, 255, 0)

    def draw_shape(self, canvas, x, y, shape, size, color):
        """Draw a shape at the specified coordinates"""
        import cv2
        import numpy as np

        x, y = int(x), int(y)

        if shape == "circle":
            cv2.circle(canvas, (x, y), size, color, -1)
            cv2.circle(canvas, (x, y), size, (255, 255, 255), 2)

        elif shape == "square":
            half_size = size
            cv2.rectangle(canvas, (x - half_size, y - half_size),
                         (x + half_size, y + half_size), color, -1)
            cv2.rectangle(canvas, (x - half_size, y - half_size),
                         (x + half_size, y + half_size), (255, 255, 255), 2)

        elif shape == "triangle":
            height = int(size * 1.732)
            half_base = size

            pts = np.array([
                [x, y - height],
                [x - half_base, y + size],
                [x + half_base, y + size]
            ], np.int32)

            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)

    def plot_coordinates(self, coordinates, image=None, point_shape="circle",
                        point_size=10, point_color="#00FF00", width=512, height=512):
        """Plot coordinates on image or blank canvas"""
        import json
        import cv2
        import numpy as np

        try:
            if not coordinates or not coordinates.strip():
                coords_list = []
            else:
                coords_list = json.loads(coordinates)
                if not isinstance(coords_list, list):
                    raise ValueError("Coordinates must be a JSON array")

            if image is not None:
                canvas = (image[0].cpu().numpy() * 255).astype(np.uint8)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            else:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

            color = self.parse_color(point_color)

            for coord in coords_list:
                if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                    x = float(coord['x'])
                    y = float(coord['y'])
                    self.draw_shape(canvas, x, y, point_shape, point_size, color)

            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas = canvas.astype(np.float32) / 255.0
            output = torch.from_numpy(canvas).unsqueeze(0)

            return (output,)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON coordinates: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Coordinate plotting failed: {str(e)}")