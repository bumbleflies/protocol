"""Tests for OCR-related tasks (UploadTask and OCRTask)."""

import logging
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest

from tasks.ocr import UploadTask, OCRTask
from tasks.task_item import FileTask, OCRBox
from tasks.ocr_provider import AssetUploader, OCRProvider


class TestUploadTask:
    """Test UploadTask processor."""

    def test_creation(self):
        """Test UploadTask can be created with uploader."""
        mock_uploader = Mock(spec=AssetUploader)
        task = UploadTask(uploader=mock_uploader)

        assert task.uploader == mock_uploader

    def test_process_uploads_image(self):
        """Test process() uploads image and sets asset_id."""
        # Create mock uploader
        mock_uploader = Mock(spec=AssetUploader)
        test_uuid = uuid4()
        mock_uploader.upload.return_value = test_uuid

        # Create processor
        processor = UploadTask(uploader=mock_uploader)

        # Create test FileTask with image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

        # Process
        result = processor.process(file_task)

        # Verify uploader was called
        mock_uploader.upload.assert_called_once()
        call_args = mock_uploader.upload.call_args
        assert call_args[0][1] == "test.jpg"  # filename argument

        # Verify asset_id was set
        assert result.asset_id == test_uuid

    def test_process_encodes_image_as_jpg(self):
        """Test process() encodes image as JPEG."""
        # Create mock uploader
        mock_uploader = Mock(spec=AssetUploader)
        mock_uploader.upload.return_value = uuid4()

        # Create processor
        processor = UploadTask(uploader=mock_uploader)

        # Create test FileTask with image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.png"), sort_key=1.0, img=img)

        # Process
        processor.process(file_task)

        # Verify upload was called with bytes
        mock_uploader.upload.assert_called_once()
        call_args = mock_uploader.upload.call_args
        assert isinstance(call_args[0][0], bytes)
        assert len(call_args[0][0]) > 0

    def test_process_logs_debug_messages(self, caplog):
        """Test process() logs debug messages."""
        # Create mock uploader
        mock_uploader = Mock(spec=AssetUploader)
        test_uuid = uuid4()
        mock_uploader.upload.return_value = test_uuid

        # Create processor
        processor = UploadTask(uploader=mock_uploader)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

        # Process with debug logging
        with caplog.at_level(logging.DEBUG):
            processor.process(file_task)

        # Verify debug messages
        assert "Uploading image for task" in caplog.text
        assert "Upload completed" in caplog.text

    def test_process_raises_on_upload_failure(self, caplog):
        """Test process() raises exception on upload failure."""
        # Create mock uploader that raises exception
        mock_uploader = Mock(spec=AssetUploader)
        mock_uploader.upload.side_effect = Exception("Upload failed")

        # Create processor
        processor = UploadTask(uploader=mock_uploader)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

        # Process should raise exception
        with pytest.raises(Exception, match="Upload failed"):
            with caplog.at_level(logging.ERROR):
                processor.process(file_task)

        # Should log the exception
        assert "UploadTask failed" in caplog.text

    def test_process_returns_same_task_object(self):
        """Test process() returns the same FileTask object."""
        # Create mock uploader
        mock_uploader = Mock(spec=AssetUploader)
        mock_uploader.upload.return_value = uuid4()

        # Create processor
        processor = UploadTask(uploader=mock_uploader)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=5.0, img=img)

        # Process
        result = processor.process(file_task)

        # Should be the same object
        assert result is file_task
        assert result.sort_key == 5.0


class TestOCRTask:
    """Test OCRTask processor."""

    def test_creation(self):
        """Test OCRTask can be created with provider."""
        mock_provider = Mock(spec=OCRProvider)
        task = OCRTask(provider=mock_provider)

        assert task.provider == mock_provider

    def test_process_performs_ocr(self):
        """Test process() performs OCR and sets ocr_boxes."""
        # Create mock provider
        mock_provider = Mock(spec=OCRProvider)
        test_boxes = [OCRBox(label="Test", confidence=0.9, x1=10, y1=10, x2=100, y2=10, x3=100, y3=50, x4=10, y4=50)]
        mock_provider.detect_text.return_value = test_boxes

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask with asset_id
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, asset_id=uuid4())

        # Process
        result = processor.process(file_task)

        # Verify provider was called
        mock_provider.detect_text.assert_called_once()
        call_args = mock_provider.detect_text.call_args
        assert call_args[0][0] == file_task.asset_id
        assert np.array_equal(call_args[0][1], img)

        # Verify ocr_boxes was set
        assert result.ocr_boxes == test_boxes

    def test_process_raises_if_no_asset_id(self):
        """Test process() raises ValueError if asset_id not set."""
        # Create mock provider
        mock_provider = Mock(spec=OCRProvider)

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask WITHOUT asset_id
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

        # Process should raise ValueError
        with pytest.raises(ValueError, match="has no asset_id"):
            processor.process(file_task)

        # Provider should not be called
        mock_provider.detect_text.assert_not_called()

    def test_process_raises_if_asset_id_is_none(self):
        """Test process() raises ValueError if asset_id is None."""
        # Create mock provider
        mock_provider = Mock(spec=OCRProvider)

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask with None asset_id
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)
        file_task.asset_id = None

        # Process should raise ValueError
        with pytest.raises(ValueError, match="has no asset_id"):
            processor.process(file_task)

    def test_process_logs_debug_messages(self, caplog):
        """Test process() logs debug messages."""
        # Create mock provider
        mock_provider = Mock(spec=OCRProvider)
        test_boxes = [OCRBox(label="Test", confidence=0.9, x1=0, y1=0, x2=10, y2=0, x3=10, y3=10, x4=0, y4=10)]
        mock_provider.detect_text.return_value = test_boxes

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, asset_id=uuid4())

        # Process with debug logging
        with caplog.at_level(logging.DEBUG):
            processor.process(file_task)

        # Verify debug messages
        assert "Performing OCR for task" in caplog.text
        assert "OCR completed" in caplog.text
        assert "found 1 text boxes" in caplog.text

    def test_process_raises_on_ocr_failure(self, caplog):
        """Test process() raises exception on OCR failure."""
        # Create mock provider that raises exception
        mock_provider = Mock(spec=OCRProvider)
        mock_provider.detect_text.side_effect = Exception("OCR failed")

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, asset_id=uuid4())

        # Process should raise exception
        with pytest.raises(Exception, match="OCR failed"):
            with caplog.at_level(logging.ERROR):
                processor.process(file_task)

        # Should log the exception
        assert "OCRTask failed" in caplog.text

    def test_process_returns_same_task_object(self):
        """Test process() returns the same FileTask object."""
        # Create mock provider
        mock_provider = Mock(spec=OCRProvider)
        mock_provider.detect_text.return_value = []

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=5.0, img=img, asset_id=uuid4())

        # Process
        result = processor.process(file_task)

        # Should be the same object
        assert result is file_task
        assert result.sort_key == 5.0

    def test_process_handles_empty_ocr_results(self):
        """Test process() handles empty OCR results."""
        # Create mock provider returning empty list
        mock_provider = Mock(spec=OCRProvider)
        mock_provider.detect_text.return_value = []

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, asset_id=uuid4())

        # Process
        result = processor.process(file_task)

        # Should set ocr_boxes to empty list
        assert result.ocr_boxes == []

    def test_process_handles_multiple_ocr_boxes(self):
        """Test process() handles multiple OCR boxes."""
        # Create mock provider with multiple boxes
        mock_provider = Mock(spec=OCRProvider)
        test_boxes = [
            OCRBox(label="Box1", confidence=0.9, x1=0, y1=0, x2=10, y2=0, x3=10, y3=10, x4=0, y4=10),
            OCRBox(label="Box2", confidence=0.85, x1=20, y1=20, x2=30, y2=20, x3=30, y3=30, x4=20, y4=30),
            OCRBox(label="Box3", confidence=0.95, x1=40, y1=40, x2=50, y2=40, x3=50, y3=50, x4=40, y4=50),
        ]
        mock_provider.detect_text.return_value = test_boxes

        # Create processor
        processor = OCRTask(provider=mock_provider)

        # Create test FileTask
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, asset_id=uuid4())

        # Process
        result = processor.process(file_task)

        # Should set all ocr_boxes
        assert len(result.ocr_boxes) == 3
        assert result.ocr_boxes == test_boxes
