import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import sys

# Self-contained SeC inference - no external path needed

from .inference.configuration_sec import SeCConfig
from .inference.modeling_sec import SeCModel
from transformers import AutoTokenizer


class SeCModelLoader:
    """
    ComfyUI node for loading SeC (Segment Concept) models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "OpenIXCLab/SeC-4B",
                    "tooltip": "HuggingFace model ID or local path to SeC model. Default is the official 4B parameter model."
                }),
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Data precision for model inference. bfloat16 recommended for best speed/quality balance."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Target device for model inference. 'auto' selects CUDA if available, otherwise CPU."
                })
            },
            "optional": {
                "use_flash_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster inference. Advanced option."
                }),
                "hydra_overrides": ("STRING", {
                    "default": "++model.non_overlap_masks=false",
                    "tooltip": "Advanced: Hydra configuration overrides (comma-separated)."
                })
            }
        }
    
    RETURN_TYPES = ("SEC_MODEL", "SEC_TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "SeC"
    TITLE = "SeC Model Loader"
    
    def load_model(self, model_path, torch_dtype, device, use_flash_attn=True, hydra_overrides="++model.non_overlap_masks=false"):
        """Load SeC model and tokenizer"""
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Parse torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16, 
            "float32": torch.float32
        }
        torch_dtype = dtype_map[torch_dtype]
        
        # Parse hydra overrides
        hydra_overrides_extra = []
        if hydra_overrides.strip():
            hydra_overrides_extra = [override.strip() for override in hydra_overrides.split(",")]
        
        try:
            # Load configuration
            config = SeCConfig.from_pretrained(model_path)
            config.hydra_overrides_extra = hydra_overrides_extra

            # Load model with proper memory management
            # For large models, use device_map to avoid OOM and BSOD
            load_kwargs = {
                "config": config,
                "torch_dtype": torch_dtype,
                "use_flash_attn": use_flash_attn,
            }

            # Use device_map for automatic memory management on CUDA
            if device == "cuda":
                # Clear CUDA cache before loading
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # Use device_map="auto" for large models to prevent OOM
                load_kwargs["device_map"] = "auto"
                load_kwargs["low_cpu_mem_usage"] = True

                print(f"Loading SeC model with device_map='auto' for safe GPU memory allocation...")
            else:
                # For CPU, load normally but with low memory mode
                load_kwargs["low_cpu_mem_usage"] = True

            model = SeCModel.from_pretrained(
                model_path,
                **load_kwargs
            ).eval()

            # Force entire model to consistent dtype to avoid mixed precision issues
            # This is critical when using device_map="auto" which can cause dtype fragmentation
            if device == "cuda" and torch_dtype != torch.float32:
                print(f"Converting entire model to {torch_dtype} to ensure dtype consistency...")
                model = model.to(dtype=torch_dtype)
            elif device != "cuda":
                # Only manually move to device if not using device_map
                model = model.to(device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Prepare model for generation
            model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

            # NUCLEAR OPTION: Register forward hooks to automatically convert all inputs to model dtype
            # This prevents ANY dtype mismatch by converting at every module boundary
            if device == "cuda" and torch_dtype != torch.float32:
                print(f"Installing dtype conversion hooks to ensure {torch_dtype} consistency...")

                def dtype_conversion_hook(module, args, kwargs):
                    """Automatically convert all tensor inputs to match module's dtype"""
                    try:
                        # Get the module's dtype from its parameters
                        module_dtype = None
                        for param in module.parameters():
                            module_dtype = param.dtype
                            break

                        if module_dtype is None:
                            return args, kwargs

                        # Skip conversion for Embedding layers - they need integer indices
                        if isinstance(module, torch.nn.Embedding):
                            return args, kwargs

                        # Convert all tensor arguments (but preserve integer dtypes)
                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                # Don't convert integer tensors (indices for embeddings, etc.)
                                if arg.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_args.append(arg)
                                elif arg.dtype != module_dtype:
                                    new_args.append(arg.to(dtype=module_dtype))
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)

                        # Convert all tensor keyword arguments (but preserve integer dtypes)
                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                # Don't convert integer tensors
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

                # Register hook on ALL modules that have parameters (Conv, Linear, etc.)
                for module in model.modules():
                    if len(list(module.parameters(recurse=False))) > 0:
                        module.register_forward_pre_hook(dtype_conversion_hook, with_kwargs=True)

                print(f"Dtype conversion hooks installed on {sum(1 for _ in model.modules())} modules")

            print(f"SeC model loaded successfully on {device}")

            return (model, tokenizer)
            
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
                "tokenizer": ("SEC_TOKENIZER", {
                    "tooltip": "SeC tokenizer loaded from SeCModelLoader node"
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
                    "default": 7,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Advanced: Frames in multimodal memory"
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
            # Parse JSON string to list of dicts
            points_list = json.loads(points_str)

            if not isinstance(points_list, list) or len(points_list) == 0:
                return None, None

            # Extract x, y coordinates from dicts
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
        # tensor shape: (batch, height, width, channels)
        images = []
        for i in range(tensor.shape[0]):
            # Convert from [0,1] to [0,255] and change to uint8
            img_array = (tensor[i] * 255).clamp(0, 255).byte().cpu().numpy()
            # Convert RGB to PIL Image
            pil_img = Image.fromarray(img_array, mode='RGB')
            images.append(pil_img)
        return images
    
    def pil_images_to_tensor(self, pil_images):
        """Convert list of PIL images to tensor"""
        if not pil_images:
            return torch.empty(0)
            
        # Convert PIL images to numpy arrays and stack
        arrays = []
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            arrays.append(arr)
        
        # Stack and convert to tensor
        tensor = torch.from_numpy(np.stack(arrays))
        return tensor
    
    def mask_to_tensor(self, mask_array):
        """Convert numpy mask to ComfyUI MASK tensor (2D grayscale)"""
        # Handle multi-dimensional arrays by taking first channel/frame
        if mask_array.ndim > 2:
            # If 3D or 4D, take first channel (assumes grayscale or first channel contains mask)
            mask_array = mask_array[..., 0] if mask_array.shape[-1] <= 4 else mask_array[0]

        # Convert to 2D float tensor [H, W] - ComfyUI MASK format
        mask_tensor = torch.from_numpy(mask_array.astype(np.float32))

        return mask_tensor
    
    def save_frames_temporarily(self, pil_images, temp_dir="/tmp/sec_frames"):
        """Save frames temporarily for video processing"""
        os.makedirs(temp_dir, exist_ok=True)
        
        # Clear existing frames
        for file in os.listdir(temp_dir):
            if file.endswith(('.jpg', '.png')):
                os.remove(os.path.join(temp_dir, file))
        
        # Save new frames
        frame_paths = []
        for i, img in enumerate(pil_images):
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            img.save(frame_path, 'JPEG', quality=95)
            frame_paths.append(frame_path)
        
        return temp_dir, frame_paths
    
    def segment_video(self, model, tokenizer, frames, positive_points="", negative_points="",
                     bbox="", input_mask=None, tracking_direction="forward",
                     annotation_frame_idx=0, object_id=1, max_frames_to_track=-1, mllm_memory_size=7):
        """Perform video object segmentation"""
        
        try:
            # Convert frames tensor to PIL images
            pil_images = self.tensor_to_pil_images(frames)
            
            # Save frames temporarily for video processing
            video_dir, frame_paths = self.save_frames_temporarily(pil_images)
            
            # Initialize inference state
            inference_state = model.grounding_encoder.init_state(video_path=video_dir)
            model.grounding_encoder.reset_state(inference_state)
            
            # Parse input prompts
            pos_points, pos_labels = self.parse_points(positive_points)
            neg_points, neg_labels = self.parse_points(negative_points)
            bbox_coords = self.parse_bbox(bbox)
            
            # Combine positive and negative points
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
            
            # Add initial annotation
            init_mask = None
            if points is not None or bbox_coords is not None:
                # Use SAM2-style prompting
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels if points is not None else None,
                    box=bbox_coords,
                )
                init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                
            elif input_mask is not None:
                # Use provided mask (MASK type is [batch, H, W])
                # Convert mask tensor to numpy
                mask_array = input_mask[0].cpu().numpy()  # Take first mask from batch [H, W]
                # MASK type is already 2D grayscale, just convert to binary
                init_mask = (mask_array > 0.5).astype(np.bool_)

                # Add mask to inference state
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    mask=init_mask,
                )

            else:
                raise ValueError("At least one visual prompt (points, bbox, or mask) must be provided for initialization")
            
            # Set tracking parameters
            if max_frames_to_track == -1:
                max_frames_to_track = len(pil_images)
            
            # Perform video propagation based on tracking direction
            video_segments = {}
            
            if tracking_direction == "bidirectional":
                # First track forward from annotation frame
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=False,
                    init_mask=init_mask,
                    tokenizer=tokenizer,
                    mllm_memory_size=mllm_memory_size,
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                
                # Reset state and track backward from annotation frame
                model.grounding_encoder.reset_state(inference_state)
                
                # Re-add initial annotation for backward tracking
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
                
                # Track backward
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=True,
                    init_mask=init_mask,
                    tokenizer=tokenizer,
                    mllm_memory_size=mllm_memory_size,
                ):
                    if out_frame_idx not in video_segments:  # Avoid overwriting forward results
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
            else:
                # Single direction tracking
                reverse = (tracking_direction == "backward")
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=reverse,
                    init_mask=init_mask,
                    tokenizer=tokenizer,
                    mllm_memory_size=mllm_memory_size,
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            
            # Convert results to ComfyUI format
            output_masks = []
            output_obj_ids = []

            for frame_idx in sorted(video_segments.keys()):
                for obj_id, mask in video_segments[frame_idx].items():
                    mask_tensor = self.mask_to_tensor(mask)
                    output_masks.append(mask_tensor)
                    output_obj_ids.append(obj_id)
            
            if not output_masks:
                # Return empty mask if no results (2D grayscale for MASK type)
                empty_mask = torch.zeros(frames.shape[1], frames.shape[2])
                output_masks = [empty_mask]
                output_obj_ids = [0]
            
            # Stack all masks
            masks_tensor = torch.stack(output_masks)
            obj_ids_tensor = torch.tensor(output_obj_ids, dtype=torch.int32)
            
            # Clean up temporary files
            import shutil
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)
            
            return (masks_tensor, obj_ids_tensor)
            
        except Exception as e:
            raise RuntimeError(f"SeC video segmentation failed: {str(e)}")