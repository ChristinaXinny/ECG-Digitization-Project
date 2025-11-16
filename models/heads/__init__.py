"""Model heads for ECG digitization."""

from .detection_head import DetectionHead, GridPointHead
from .segmentation_head import SegmentationHead
from .regression_head import RegressionHead
from .classification_head import (
    ClassificationHead, OrientationClassificationHead,
    LeadClassificationHead, MultiLabelClassificationHead
)

__all__ = [
    "DetectionHead",
    "GridPointHead",
    "SegmentationHead",
    "RegressionHead",
    "ClassificationHead",
    "OrientationClassificationHead",
    "LeadClassificationHead",
    "MultiLabelClassificationHead"
]