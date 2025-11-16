"""Model definitions for ECG digitization."""

from .base_model import BaseModel
from .stage0_model import Stage0Net
from .stage1_model import Stage1Net
from .stage2_model import Stage2Net

from .heads.detection_head import DetectionHead, GridPointHead
from .heads.regression_head import RegressionHead
from .heads.segmentation_head import SegmentationHead

__all__ = [
    "BaseModel",
    "Stage0Net",
    "Stage1Net",
    "Stage2Net",
    "DetectionHead",
    "GridPointHead",
    "RegressionHead",
    "SegmentationHead"
]