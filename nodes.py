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
    from huggingface_hub import snapshot_download

    sams_paths = folder_paths.get_folder_paths("sams")
    destination = os.path.join(sams_paths[0], "SeC-4B")

    print("=" * 80)
    print("SeC-4B model not found. Downloading from HuggingFace...")
    print(f"Repository: OpenIXCLab/SeC-4B")
    print(f"Destination: {destination}")
    print("=" * 80)

    # Create directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    snapshot_download(
        repo_id="OpenIXCLab/SeC-4B",
        local_dir=destination,
        local_dir_use_symlinks=False
    )

    print("=" * 80)
    print(f"✓ SeC-4B model downloaded successfully!")
    print(f"✓ Location: {destination}")
    print("=" * 80)

    return destination


class SeCModelLoader:
    """
    ComfyUI node for loading SeC (Segment Concept) models
    """

    _cached_model = None
    _cache_config = None

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically build device list based on available GPUs
        device_choices = ["auto", "cpu"]

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                device_choices.append(f"gpu{i}")

            if gpu_count > 1:
                device_choices.append("multi-gpu")

        return {
            "required": {
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Data precision for model inference. bfloat16 recommended for best speed/quality balance."
                }),
                "device": (device_choices, {
                    "default": "auto",
                    "tooltip": "Device: auto (gpu0 if available, else CPU), cpu, gpu0-N (specific GPU), multi-gpu (split across all GPUs)"
                })
            },
            "optional": {
                "use_flash_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster inference."
                }),
                "allow_mask_overlap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow tracked objects to overlap. Disable for strictly separate objects."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Memory: Cache model in memory for reuse across multiple nodes. Disable to load fresh each time (slower but frees memory between uses)."
                })
            }
        }
    
    RETURN_TYPES = ("SEC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SeC"
    TITLE = "SeC Model Loader"
    
    def load_model(self, torch_dtype, device, use_flash_attn=True, allow_mask_overlap=True, cache_model=True):
        """Load SeC model and tokenizer"""

        # Create config tuple for cache comparison
        current_config = (torch_dtype, device, use_flash_attn, allow_mask_overlap)

        # Check if we can use cached model
        if cache_model and SeCModelLoader._cached_model is not None and SeCModelLoader._cache_config == current_config:
            print("=" * 80)
            print("✓ Using cached SeC model")
            print("=" * 80)
            return (SeCModelLoader._cached_model,)

        # Clear cache if cache_model is False
        if not cache_model and SeCModelLoader._cached_model is not None:
            print("Clearing model cache...")
            SeCModelLoader._cached_model = None
            SeCModelLoader._cache_config = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        original_device = device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            # Map gpu0 -> cuda:0, gpu1 -> cuda:1, etc.
            gpu_num = device[3:]  # Extract number after "gpu"
            device = f"cuda:{gpu_num}"
        elif device == "multi-gpu":
            device = "cuda"  # Will use device_map='auto' to split

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[torch_dtype]

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
            if original_device == "multi-gpu":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                load_kwargs["device_map"] = "auto"  # Split across all GPUs
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"Loading SeC model with device_map='auto' (multi-GPU)...")
            elif device.startswith("cuda:"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                load_kwargs["device_map"] = {"": device}  # Force to specific GPU
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"Loading SeC model on {device}...")
            else:
                # CPU mode
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"Loading SeC model on CPU...")

            model = SeCModel.from_pretrained(model_path, **load_kwargs).eval()

            if device.startswith("cuda") and torch_dtype != torch.float32 and original_device != "multi-gpu":
                print(f"Converting model to {torch_dtype}...")
                model = model.to(dtype=torch_dtype)
            elif device == "cpu":
                model = model.to(device)

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

            # Cache model if caching is enabled
            if cache_model:
                SeCModelLoader._cached_model = model
                SeCModelLoader._cache_config = current_config
                print("✓ Model cached for reuse")

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
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Advanced: Frames in multimodal memory. Lower values use less memory."
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
    
    def parse_points(self, points_str):
        """Parse point coordinates from JSON string: '[{\"x\": 63, \"y\": 782}]'"""
        import json

        if not points_str or not points_str.strip():
            return None, None

        try:
            points_list = json.loads(points_str)

            if not isinstance(points_list, list) or len(points_list) == 0:
                return None, None

            points = []
            for point_dict in points_list:
                if isinstance(point_dict, dict) and 'x' in point_dict and 'y' in point_dict:
                    points.append([float(point_dict['x']), float(point_dict['y'])])

            if not points:
                return None, None

            return np.array(points, dtype=np.float32), np.ones(len(points), dtype=np.int32)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing points: {e}")
            return None, None
    
    def parse_bbox(self, bbox_str):
        """Parse bounding box from string"""
        if not bbox_str.strip():
            return None
            
        coords = list(map(float, bbox_str.strip().split(',')))
        if len(coords) != 4:
            return None
            
        return np.array(coords, dtype=np.float32)
    
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
    
    def save_frames_temporarily(self, pil_images, temp_dir="/tmp/sec_frames"):
        """Save frames temporarily for video processing"""
        os.makedirs(temp_dir, exist_ok=True)

        for file in os.listdir(temp_dir):
            if file.endswith(('.jpg', '.png')):
                os.remove(os.path.join(temp_dir, file))

        frame_paths = []
        for i, img in enumerate(pil_images):
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            img.save(frame_path, 'JPEG', quality=95)
            frame_paths.append(frame_path)

        return temp_dir, frame_paths
    
    def segment_video(self, model, frames, positive_points="", negative_points="",
                     bbox="", input_mask=None, tracking_direction="forward",
                     annotation_frame_idx=0, object_id=1, max_frames_to_track=-1, mllm_memory_size=5,
                     offload_video_to_cpu=False):
        """Perform video object segmentation"""

        try:
            pil_images = self.tensor_to_pil_images(frames)
            video_dir, frame_paths = self.save_frames_temporarily(pil_images)

            # Automatically set offload_state_to_cpu based on model device
            offload_state_to_cpu = str(model.device) == "cpu"

            inference_state = model.grounding_encoder.init_state(
                video_path=video_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu
            )
            model.grounding_encoder.reset_state(inference_state)

            pos_points, pos_labels = self.parse_points(positive_points)
            neg_points, neg_labels = self.parse_points(negative_points)
            bbox_coords = self.parse_bbox(bbox)

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
                raise ValueError("At least one visual prompt (points, bbox, or mask) must be provided")

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

            import shutil
            import gc
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Unload model if not cached
            # If model is cached, SeCModelLoader will provide same instance to next node
            # If not cached, each node gets fresh model that can be safely deleted
            if model is not SeCModelLoader._cached_model:
                try:
                    # Delete all model components to free memory completely
                    if hasattr(model, 'grounding_encoder'):
                        del model.grounding_encoder
                    if hasattr(model, 'vision_model'):
                        del model.vision_model
                    if hasattr(model, 'language_model'):
                        del model.language_model
                    if hasattr(model, 'tokenizer'):
                        del model.tokenizer

                    # Delete the model itself
                    del model

                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print("✓ Model unloaded from memory (not cached)")
                except Exception as e:
                    print(f"Warning: Model unload failed: {e}")
            else:
                print("✓ Model retained in cache for reuse")

            return (masks_tensor, obj_ids_tensor)

        except Exception as e:
            raise RuntimeError(f"SeC video segmentation failed: {str(e)}")


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