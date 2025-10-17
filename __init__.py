"""
ComfyUI Custom Nodes for SeC (Segment Concept) Video Object Segmentation
"""

import os
import folder_paths

# Register 'sams' folder type for SeC models
# This integrates with ComfyUI's folder_paths system and extra_model_paths.yaml
models_dir = folder_paths.models_dir
sams_dir = os.path.join(models_dir, "sams")
folder_paths.add_model_folder_path("sams", sams_dir)

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "SeCModelLoader": SeCModelLoader,
    "SeCVideoSegmentation": SeCVideoSegmentation,
    "CoordinatePlotter": CoordinatePlotter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeCModelLoader": "SeC Model Loader Ultra-Fast",
    "SeCVideoSegmentation": "SeC Video Segmentation Ultra-Fast",
    "CoordinatePlotter": "Coordinate Plotter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]