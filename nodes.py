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
                "use_flash_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster and more memory-efficient attention computation."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Target device for model inference. 'auto' selects CUDA if available, otherwise CPU."
                }),
                "hydra_overrides": ("STRING", {
                    "default": "++model.non_overlap_masks=false",
                    "tooltip": "Hydra configuration overrides (comma-separated). Controls model behavior like mask overlapping."
                }),
                "grounding_maskmem_num": ("INT", {
                    "default": 22,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of memory frames stored by the grounding encoder for temporal consistency."
                })
            }
        }
    
    RETURN_TYPES = ("SEC_MODEL", "SEC_TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "SeC"
    TITLE = "SeC Model Loader"
    
    def load_model(self, model_path, torch_dtype, use_flash_attn, device, hydra_overrides, grounding_maskmem_num):
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
            config.grounding_maskmem_num = grounding_maskmem_num
            
            # Load model
            model = SeCModel.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch_dtype,
                use_flash_attn=use_flash_attn
            ).eval().to(device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Prepare model for generation
            model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)
            
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
                    "tooltip": "Sequential video frames as IMAGE tensor batch. Should contain all frames of the video in temporal order."
                }),
                "annotation_frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index where initial annotation/prompt is applied. Usually the first frame (0) or a clear frame."
                }),
                "object_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Unique identifier for this object. Use different IDs when tracking multiple objects simultaneously."
                })
            },
            "optional": {
                "positive_points": ("STRING", {
                    "default": "",
                    "tooltip": "Positive click coordinates as 'x1,y1;x2,y2;x3,y3'. Points that belong to the target object."
                }),
                "negative_points": ("STRING", {
                    "default": "",
                    "tooltip": "Negative click coordinates as 'x1,y1;x2,y2'. Points that do NOT belong to the target object."
                }),
                "bbox": ("STRING", {
                    "default": "",
                    "tooltip": "Bounding box as 'x_min,y_min,x_max,y_max'. Rectangular region containing the target object."
                }),
                "input_mask": ("IMAGE", {
                    "tooltip": "Binary mask image for precise object initialization. White pixels indicate the target object."
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Natural language description of the object to segment (e.g., 'person wearing red shirt', 'flying bird')."
                }),
                "start_frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame index for temporal propagation. Default 0 starts from the beginning."
                }),
                "max_frames_to_track": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Maximum number of frames to process (-1 for all frames). Useful for limiting computation."
                }),
                "tracking_direction": (["forward", "backward", "bidirectional"], {
                    "default": "forward",
                    "tooltip": "Tracking direction: 'forward' (annotation→end), 'backward' (annotation→start), 'bidirectional' (both directions from annotation frame)"
                }),
                "mllm_memory_size": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Number of recent frames stored in multimodal memory for semantic reasoning and scene understanding."
                }),
                "output_stride": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Output every N-th frame to reduce output size. Set to 1 for all frames, 2 for every other frame, etc."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("masks", "object_ids") 
    FUNCTION = "segment_video"
    CATEGORY = "SeC"
    TITLE = "SeC Video Segmentation"
    DESCRIPTION = ("Concept-driven video object segmentation that intelligently combines visual features with semantic reasoning. "
                   "Supports points, bounding boxes, masks, and text prompts. Adapts computational effort based on scene complexity.")
    
    def parse_points(self, points_str):
        """Parse point coordinates from string"""
        if not points_str.strip():
            return None, None
            
        points = []
        for point_pair in points_str.split(';'):
            if point_pair.strip():
                x, y = map(float, point_pair.strip().split(','))
                points.append([x, y])
        
        if not points:
            return None, None
            
        return np.array(points, dtype=np.float32), np.ones(len(points), dtype=np.int32)
    
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
        """Convert numpy mask to tensor"""
        # Ensure mask is in correct format
        if mask_array.ndim == 2:
            mask_array = mask_array[..., np.newaxis]
        
        # Convert boolean mask to float and repeat for RGB channels
        mask_tensor = torch.from_numpy(mask_array.astype(np.float32))
        if mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor.repeat(1, 1, 3)
        
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
    
    def segment_video(self, model, tokenizer, frames, annotation_frame_idx, object_id,
                     positive_points="", negative_points="", bbox="", input_mask=None, 
                     text_prompt="", start_frame_idx=0, max_frames_to_track=-1,
                     tracking_direction="forward", mllm_memory_size=7, output_stride=1):
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
                # Use provided mask
                # Convert mask tensor to numpy
                mask_array = input_mask[0].cpu().numpy()  # Take first image if batch
                if mask_array.shape[-1] == 3:  # RGB mask
                    mask_array = mask_array[:, :, 0] > 0.5  # Convert to binary
                init_mask = mask_array.astype(np.bool_)
                
                # Add mask to inference state (this might need adjustment based on SeC API)
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    mask=init_mask,
                )
                
            else:
                raise ValueError("At least one prompt type (points, bbox, mask, or text) must be provided")
            
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
                    start_frame_idx=annotation_frame_idx if start_frame_idx == 0 else start_frame_idx,
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
                if frame_idx % output_stride == 0:  # Apply output stride
                    for obj_id, mask in video_segments[frame_idx].items():
                        mask_tensor = self.mask_to_tensor(mask)
                        output_masks.append(mask_tensor)
                        output_obj_ids.append(obj_id)
            
            if not output_masks:
                # Return empty mask if no results
                empty_mask = torch.zeros(frames.shape[1], frames.shape[2], 3)
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