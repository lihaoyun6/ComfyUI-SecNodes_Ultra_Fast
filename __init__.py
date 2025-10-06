"""
ComfyUI Custom Nodes for SeC (Segment Concept) Video Object Segmentation
"""

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "SeCModelLoader": SeCModelLoader,
    "SeCVideoSegmentation": SeCVideoSegmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeCModelLoader": "SeC Model Loader",
    "SeCVideoSegmentation": "SeC Video Segmentation",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]