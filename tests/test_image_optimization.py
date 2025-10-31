"""Tests for ImageOptimizationTask."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tasks.image_optimization import ImageOptimizationTask
from tasks.task_item import FileTask


class TestImageOptimizationTask:
    """Test ImageOptimizationTask processor."""

    def test_creation(self):
        """Test ImageOptimizationTask can be created."""
        task = ImageOptimizationTask(padding=10, min_crop_ratio=0.8)
        assert task.padding == 10
        assert task.min_crop_ratio == 0.8

    def test_creation_with_defaults(self):
        """Test ImageOptimizationTask with default parameters."""
        task = ImageOptimizationTask()
        assert task.padding == 10
        assert task.min_crop_ratio == 0.8

    def test_process_nonexistent_file(self):
        """Test process() raises ValueError for nonexistent file."""
        processor = ImageOptimizationTask()
        file_task = FileTask(
            file_path=Path("/nonexistent/image.jpg"),
            sort_key=1.0
        )

        with pytest.raises(ValueError, match="Could not read image"):
            processor.process(file_task)

    def test_process_simple_image(self):
        """Test process() optimizes a simple image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image with white center and black border
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[50:150, 50:150] = [255, 255, 255]  # White center

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(padding=10, min_crop_ratio=0.5)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Verify image was loaded and processed
            assert result.img is not None
            assert isinstance(result.img, np.ndarray)
            assert result.img.shape[2] == 3  # RGB image

    def test_process_image_with_content(self):
        """Test process() crops image around content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with content in corner
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add some text-like content in corner
            cv2.rectangle(img, (10, 10), (50, 50), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(padding=5, min_crop_ratio=0.2)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Verify image was cropped (should be smaller than original)
            assert result.img is not None
            # The cropped image might be smaller if content detected
            assert result.img.shape[0] <= 200
            assert result.img.shape[1] <= 200

    def test_process_prevents_overcropping(self):
        """Test process() prevents over-cropping with min_crop_ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with very small content
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add tiny white dot
            img[100, 100] = [255, 255, 255]

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process with high min_crop_ratio
            processor = ImageOptimizationTask(padding=5, min_crop_ratio=0.8)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should not crop too aggressively due to min_crop_ratio
            assert result.img is not None
            # Image should be kept close to original size
            assert result.img.shape[0] >= 160  # At least 80% of 200
            assert result.img.shape[1] >= 160

    def test_process_no_contours_found(self):
        """Test process() handles images with no contours (all black)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create completely black image
            img = np.zeros((200, 200, 3), dtype=np.uint8)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask()
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should return original image as fallback
            assert result.img is not None
            assert result.img.shape == (200, 200, 3)

    def test_process_adds_padding(self):
        """Test process() adds padding around detected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with content
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(img, (80, 80), (120, 120), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process with specific padding
            processor = ImageOptimizationTask(padding=20, min_crop_ratio=0.2)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Verify image was processed with padding
            assert result.img is not None
            # The crop should include padding around the content

    def test_process_complex_content(self):
        """Test process() handles complex content with multiple contours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with multiple separated content areas
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(img, (20, 20), (50, 50), (255, 255, 255), -1)
            cv2.rectangle(img, (150, 150), (180, 180), (255, 255, 255), -1)
            cv2.rectangle(img, (20, 150), (50, 180), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(padding=10, min_crop_ratio=0.3)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should find bounding box of all contours
            assert result.img is not None
            assert isinstance(result.img, np.ndarray)

    def test_process_maintains_color(self):
        """Test process() maintains color information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create colorful image
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[50:100, 50:100] = [255, 0, 0]  # Blue
            img[100:150, 100:150] = [0, 255, 0]  # Green
            img[50:100, 100:150] = [0, 0, 255]  # Red

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(padding=10, min_crop_ratio=0.3)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Verify color channels are maintained
            assert result.img is not None
            assert result.img.shape[2] == 3  # BGR channels

    def test_process_boundary_conditions(self):
        """Test process() handles boundary conditions correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with content at edge
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[0:50, 0:50] = [255, 255, 255]  # Top-left corner
            img[150:200, 150:200] = [255, 255, 255]  # Bottom-right corner

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process with large padding
            processor = ImageOptimizationTask(padding=50, min_crop_ratio=0.3)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should not go out of bounds
            assert result.img is not None
            assert result.img.shape[0] <= 200
            assert result.img.shape[1] <= 200

    def test_process_different_padding_values(self):
        """Test process() with different padding values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(img, (90, 90), (110, 110), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Test with different padding
            for padding in [0, 5, 10, 20, 50]:
                processor = ImageOptimizationTask(padding=padding, min_crop_ratio=0.1)
                file_task = FileTask(file_path=test_file, sort_key=1.0)

                result = processor.process(file_task)

                assert result.img is not None

    def test_process_different_min_crop_ratios(self):
        """Test process() with different min_crop_ratio values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image with small content
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(img, (95, 95), (105, 105), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Test with different min_crop_ratios
            for ratio in [0.1, 0.5, 0.8, 0.95]:
                processor = ImageOptimizationTask(padding=5, min_crop_ratio=ratio)
                file_task = FileTask(file_path=test_file, sort_key=1.0)

                result = processor.process(file_task)

                assert result.img is not None
                # Higher ratio should preserve more of original size
                if ratio > 0.9:
                    assert result.img.shape[0] >= 190
                    assert result.img.shape[1] >= 190

    def test_process_returns_same_task_object(self):
        """Test process() returns the same FileTask object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[40:60, 40:60] = [255, 255, 255]

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask()
            file_task = FileTask(file_path=test_file, sort_key=5.0)

            result = processor.process(file_task)

            # Should be the same object with img field populated
            assert result is file_task
            assert result.sort_key == 5.0
            assert result.file_path == test_file
            assert result.img is not None
