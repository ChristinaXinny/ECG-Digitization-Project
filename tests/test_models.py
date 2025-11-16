#!/usr/bin/env python3
"""Model tests for ECG Digitization Project."""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import Stage0Net
from models.heads.detection_head import DetectionHead, GridPointHead, MultiScaleDetectionHead
from models.heads.segmentation_head import SegmentationHead
from models.heads.regression_head import RegressionHead
from models.heads.classification_head import (
    ClassificationHead, OrientationClassificationHead,
    LeadClassificationHead, MultiLabelClassificationHead
)


class TestDetectionHeads(unittest.TestCase):
    """Test detection head implementations."""

    def test_basic_detection_head(self):
        """Test basic detection head."""
        head = DetectionHead(
            in_channels=256,
            num_classes=14,
            activation='softmax'
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))
        self.assertAlmostEqual(output.sum(dim=1).mean().item(), 1.0, places=2)  # Softmax sum

    def test_detection_head_sigmoid(self):
        """Test detection head with sigmoid activation."""
        head = DetectionHead(
            in_channels=256,
            num_classes=14,
            activation='sigmoid'
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_detection_head_no_activation(self):
        """Test detection head with no activation."""
        head = DetectionHead(
            in_channels=256,
            num_classes=14,
            activation='none'
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))

    def test_detection_head_with_attention(self):
        """Test detection head with self-attention."""
        head = DetectionHead(
            in_channels=256,
            num_classes=14,
            use_attention=True
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))

    def test_grid_point_head(self):
        """Test grid point detection head."""
        head = GridPointHead(
            in_channels=256,
            num_classes=1,
            use_coordinate_encoding=True
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 1, 32, 32))

    def test_multi_scale_detection_head(self):
        """Test multi-scale detection head."""
        in_channels_list = [64, 128, 256]
        head = MultiScaleDetectionHead(
            in_channels_list=in_channels_list,
            num_classes=14,
            feature_fusion='concat'
        )

        # Create multi-scale features
        features = [
            torch.randn(2, 64, 64, 64),
            torch.randn(2, 128, 32, 32),
            torch.randn(2, 256, 16, 16)
        ]

        outputs = head(features)

        self.assertIn('scale_0', outputs)
        self.assertIn('scale_1', outputs)
        self.assertIn('scale_2', outputs)
        self.assertIn('fused', outputs)


class TestSegmentationHead(unittest.TestCase):
    """Test segmentation head."""

    def test_segmentation_head(self):
        """Test basic segmentation head."""
        head = SegmentationHead(
            in_channels=256,
            num_classes=14,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))

    def test_segmentation_head_with_decoder(self):
        """Test segmentation head with decoder."""
        head = SegmentationHead(
            in_channels=256,
            num_classes=14,
            use_decoder=True,
            decoder_channels=[128, 64]
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14, 32, 32))


class TestRegressionHead(unittest.TestCase):
    """Test regression head."""

    def test_regression_head(self):
        """Test basic regression head."""
        head = RegressionHead(
            in_channels=256,
            num_outputs=8,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 8))

    def test_regression_head_with_hidden_layer(self):
        """Test regression head with hidden dimension."""
        head = RegressionHead(
            in_channels=256,
            num_outputs=4,
            hidden_dim=128,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 4))

    def test_regression_head_single_output(self):
        """Test regression head with single output."""
        head = RegressionHead(
            in_channels=256,
            num_outputs=1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 1))


class TestClassificationHeads(unittest.TestCase):
    """Test classification head implementations."""

    def test_basic_classification_head(self):
        """Test basic classification head."""
        head = ClassificationHead(
            in_channels=256,
            num_classes=8,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 8))

    def test_orientation_classification_head(self):
        """Test orientation classification head."""
        head = OrientationClassificationHead(
            in_channels=256,
            num_orientations=8,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 8))

    def test_lead_classification_head(self):
        """Test lead classification head."""
        head = LeadClassificationHead(
            in_channels=256,
            num_leads=12,  # Standard 12-lead ECG
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 12))

    def test_multilabel_classification_head(self):
        """Test multi-label classification head."""
        head = MultiLabelClassificationHead(
            in_channels=256,
            num_classes=14,
            dropout=0.1
        )

        x = torch.randn(2, 256, 32, 32)
        output = head(x)

        self.assertEqual(output.shape, (2, 14))


class TestStage0Net(unittest.TestCase):
    """Test Stage0Net model."""

    def setUp(self):
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'INPUT_SIZE': [512, 512],
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8,
                'DECODER': {'ENABLED': True},
                'ATTENTION': {'ENABLED': True},
                'HEADS': {'REDUCED_DIM': False}
            }
        }

    def test_model_creation(self):
        """Test Stage0Net model creation."""
        model = Stage0Net(self.config)

        # Check model has required components
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.decoder)
        self.assertIsNotNone(model.marker_head)
        self.assertIsNotNone(model.orientation_head)

    def test_model_parameter_count(self):
        """Test model parameter count."""
        model = Stage0Net(self.config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 1000000)  # Should have > 1M parameters
        self.assertGreater(trainable_params, 1000000)

    def test_model_forward(self):
        """Test forward pass through Stage0Net."""
        model = Stage0Net(self.config)

        # Create input batch
        batch = {
            'image': torch.randn(2, 3, 512, 512),
            'marker': torch.randint(0, 14, (2, 512, 512)),
            'orientation': torch.randint(0, 8, (2,))
        }

        # Forward pass
        outputs = model(batch)

        self.assertIn('marker', outputs)
        self.assertIn('orientation', outputs)
        self.assertEqual(outputs['marker'].shape, (2, 14, 512, 512))
        self.assertEqual(outputs['orientation'].shape, (2, 8))

    def test_model_forward_without_labels(self):
        """Test forward pass without labels (inference mode)."""
        model = Stage0Net(self.config)

        # Create input batch without labels
        batch = {
            'image': torch.randn(2, 3, 512, 512)
        }

        # Forward pass should still work
        outputs = model(batch)

        self.assertIn('marker', outputs)
        self.assertIn('orientation', outputs)

    def test_model_different_input_sizes(self):
        """Test model with different input sizes."""
        for input_size in [[256, 256], [512, 512], [1024, 1024]]:
            with self.subTest(input_size=input_size):
                config = self.config.copy()
                config['MODEL']['INPUT_SIZE'] = input_size

                model = Stage0Net(config)

                batch = {
                    'image': torch.randn(1, 3, *input_size),
                    'marker': torch.randint(0, 14, (1, *input_size)),
                    'orientation': torch.randint(0, 8, (1,))
                }

                outputs = model(batch)

                self.assertEqual(outputs['marker'].shape[2:], tuple(input_size))
                self.assertEqual(outputs['orientation'].shape[1], 8)

    def test_model_config_variations(self):
        """Test model with different configuration variations."""
        configs = [
            # Minimal configuration
            {
                'MODEL': {
                    'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                    'NUM_MARKER_CLASSES': 14,
                    'NUM_ORIENTATION_CLASSES': 8
                }
            },
            # Configuration with decoder disabled
            {
                'MODEL': {
                    'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                    'NUM_MARKER_CLASSES': 14,
                    'NUM_ORIENTATION_CLASSES': 8,
                    'DECODER': {'ENABLED': False},
                    'ATTENTION': {'ENABLED': False}
                }
            }
        ]

        for config in configs:
            with self.subTest(config=config):
                model = Stage0Net(config)

                # Set default input size if not specified
                input_size = config['MODEL'].get('INPUT_SIZE', [512, 512])

                batch = {
                    'image': torch.randn(1, 3, *input_size),
                    'marker': torch.randint(0, 14, (1, *input_size)),
                    'orientation': torch.randint(0, 8, (1,))
                }

                outputs = model(batch)

                self.assertIn('marker', outputs)
                self.assertIn('orientation', outputs)


class TestModelGradientFlow(unittest.TestCase):
    """Test gradient flow through model components."""

    def setUp(self):
        self.config = {
            'MODEL': {
                'BACKBONE': {'NAME': 'resnet18', 'PRETRAINED': False},
                'NUM_MARKER_CLASSES': 14,
                'NUM_ORIENTATION_CLASSES': 8
            }
        }

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = Stage0Net(self.config)

        # Create input batch
        batch = {
            'image': torch.randn(2, 3, 256, 256, requires_grad=True),
            'marker': torch.randint(0, 14, (2, 256, 256)),
            'orientation': torch.randint(0, 8, (2,))
        }

        # Forward pass
        outputs = model(batch)

        # Create dummy loss
        marker_loss = nn.CrossEntropyLoss()(
            outputs['marker'], batch['marker']
        )
        orientation_loss = nn.CrossEntropyLoss()(
            outputs['orientation'], batch['orientation']
        )
        total_loss = marker_loss + orientation_loss

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")


if __name__ == '__main__':
    unittest.main()