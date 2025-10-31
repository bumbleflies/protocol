"""
Tests for OCR provider abstraction (DIP compliance).
"""

import pytest
import uuid
import numpy as np
from typing import List

from tasks.ocr_provider import AssetUploader, OCRProvider
from tasks.task_item import OCRBox


class MockAssetUploader(AssetUploader):
    """Mock uploader for testing."""

    def __init__(self):
        self.uploaded_images = []

    def upload(self, image_bytes: bytes, description: str) -> uuid.UUID:
        """Mock upload that stores data."""
        self.uploaded_images.append((image_bytes, description))
        return uuid.uuid4()


class MockOCRProvider(OCRProvider):
    """Mock OCR provider for testing."""

    def __init__(self, mock_boxes: List[OCRBox] = None):
        self.mock_boxes = mock_boxes or []
        self.detect_calls = []

    def detect_text(self, asset_id: uuid.UUID, image: np.ndarray) -> List[OCRBox]:
        """Mock OCR that returns predefined boxes."""
        self.detect_calls.append((asset_id, image))
        return self.mock_boxes


class TestOCRProviderAbstraction:
    """Test OCR provider abstraction and DIP compliance."""

    def test_asset_uploader_interface(self):
        """Test that AssetUploader interface can be implemented."""
        uploader = MockAssetUploader()
        assert isinstance(uploader, AssetUploader)

    def test_mock_uploader_uploads(self):
        """Test mock uploader functionality."""
        uploader = MockAssetUploader()
        image_bytes = b"fake image data"

        asset_id = uploader.upload(image_bytes, "test.jpg")

        assert isinstance(asset_id, uuid.UUID)
        assert len(uploader.uploaded_images) == 1
        assert uploader.uploaded_images[0][0] == image_bytes
        assert uploader.uploaded_images[0][1] == "test.jpg"

    def test_ocr_provider_interface(self):
        """Test that OCRProvider interface can be implemented."""
        provider = MockOCRProvider()
        assert isinstance(provider, OCRProvider)

    def test_mock_ocr_provider_detects_text(self):
        """Test mock OCR provider functionality."""
        mock_boxes = [
            OCRBox(label="Hello", x1=0, y1=0, x2=10, y2=0, x3=10, y3=10, x4=0, y4=10, confidence=0.9),
            OCRBox(label="World", x1=20, y1=0, x2=30, y2=0, x3=30, y3=10, x4=20, y4=10, confidence=0.85),
        ]

        provider = MockOCRProvider(mock_boxes)
        test_asset_id = uuid.uuid4()
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = provider.detect_text(test_asset_id, test_image)

        assert len(result) == 2
        assert result[0].label == "Hello"
        assert result[1].label == "World"
        assert len(provider.detect_calls) == 1

    def test_dip_compliance_depend_on_abstraction(self):
        """
        Test DIP: High-level code depends on abstraction, not concrete implementation.
        This demonstrates Dependency Inversion Principle compliance.
        """

        def process_with_providers(uploader: AssetUploader, ocr: OCRProvider, image_bytes: bytes) -> List[OCRBox]:
            """High-level function that depends on abstractions."""
            # Upload image
            asset_id = uploader.upload(image_bytes, "test.jpg")

            # Perform OCR
            fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
            boxes = ocr.detect_text(asset_id, fake_image)

            return boxes

        # Can use mock implementations
        mock_uploader = MockAssetUploader()
        mock_boxes = [OCRBox(label="Test", x1=0, y1=0, x2=10, y2=0, x3=10, y3=10, x4=0, y4=10, confidence=0.9)]
        mock_ocr = MockOCRProvider(mock_boxes)

        result = process_with_providers(mock_uploader, mock_ocr, b"test image")

        assert len(result) == 1
        assert result[0].label == "Test"
        assert len(mock_uploader.uploaded_images) == 1
        assert len(mock_ocr.detect_calls) == 1

    def test_provider_swapping(self):
        """
        Test that providers can be swapped without changing high-level code.
        This demonstrates the power of dependency injection.
        """

        class AlternativeUploader(AssetUploader):
            """Alternative uploader implementation."""

            def upload(self, image_bytes: bytes, description: str) -> uuid.UUID:
                # Different implementation
                return uuid.UUID("12345678-1234-5678-1234-567812345678")

        def use_uploader(uploader: AssetUploader) -> uuid.UUID:
            """Function that uses an uploader."""
            return uploader.upload(b"data", "file.jpg")

        # Can use mock
        mock = MockAssetUploader()
        result1 = use_uploader(mock)
        assert isinstance(result1, uuid.UUID)

        # Can swap to alternative without changing use_uploader
        alternative = AlternativeUploader()
        result2 = use_uploader(alternative)
        assert result2 == uuid.UUID("12345678-1234-5678-1234-567812345678")
