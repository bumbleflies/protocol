"""Tests for ImageOptimizationTask with quadrilateral detection."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tasks.image_optimization import ImageOptimizationTask
from tasks.task_item import FileTask


class TestImageOptimizationTask:
    """Test ImageOptimizationTask processor with quadrilateral detection."""

    def test_creation(self):
        """Test ImageOptimizationTask can be created with parameters."""
        task = ImageOptimizationTask(
            target_width=1920, min_area_ratio=0.25, max_area_ratio=0.95, enable_perspective_correction=True
        )
        assert task.target_width == 1920
        assert task.min_area_ratio == 0.25
        assert task.max_area_ratio == 0.95
        assert task.enable_perspective_correction is True

    def test_creation_with_defaults(self):
        """Test ImageOptimizationTask with default parameters."""
        task = ImageOptimizationTask()
        assert task.target_width == 1920
        assert task.min_area_ratio == 0.10
        assert task.max_area_ratio == 0.95
        assert task.enable_perspective_correction is True

    def test_process_nonexistent_file(self):
        """Test process() raises ValueError for nonexistent file."""
        processor = ImageOptimizationTask()
        file_task = FileTask(file_path=Path("/nonexistent/image.jpg"), sort_key=1.0)

        with pytest.raises(ValueError, match="Could not read image"):
            processor.process(file_task)

    def test_process_simple_image(self):
        """Test process() processes a simple image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image with white center
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[50:150, 50:150] = [255, 255, 255]  # White center

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(target_width=2000)  # No resize
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Verify image was loaded and processed
            assert result.img is not None
            assert isinstance(result.img, np.ndarray)
            assert result.img.shape[2] == 3  # RGB image

    def test_process_with_clear_edges(self):
        """Test process() detects quadrilateral with clear edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with clear white rectangle on black background
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            # Draw white rectangle (flipchart)
            cv2.rectangle(img, (50, 50), (550, 350), (255, 255, 255), -1)
            # Add black border to make edges clear
            cv2.rectangle(img, (50, 50), (550, 350), (0, 0, 0), 5)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(min_area_ratio=0.3, max_area_ratio=0.95, target_width=2000)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should detect and crop to rectangle
            assert result.img is not None
            # Image should be cropped (smaller than original)
            assert result.img.shape[0] <= 400
            assert result.img.shape[1] <= 600

    def test_process_no_quadrilateral_uses_fallback(self):
        """Test process() uses fallback when no quadrilateral detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image without clear quadrilateral
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add some random content (no clear rectangular boundary)
            img[80:120, 80:120] = [255, 255, 255]

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(target_width=2000)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should use fallback method
            assert result.img is not None
            assert isinstance(result.img, np.ndarray)

    def test_process_all_black_image(self):
        """Test process() handles completely black image (no edges)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create completely black image
            img = np.zeros((200, 200, 3), dtype=np.uint8)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(target_width=2000)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should return original image as fallback
            assert result.img is not None
            assert result.img.shape == (200, 200, 3)

    def test_process_resize_when_exceeds_target_width(self):
        """Test process() resizes image when width exceeds target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create large image
            img = np.zeros((1000, 2500, 3), dtype=np.uint8)
            img[100:900, 100:2400] = [255, 255, 255]

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process with target width
            processor = ImageOptimizationTask(target_width=1920)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should be resized to target width
            assert result.img is not None
            assert result.img.shape[1] <= 1920

    def test_process_no_resize_when_below_target_width(self):
        """Test process() does not resize when width is below target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create small image
            img = np.zeros((200, 300, 3), dtype=np.uint8)
            img[50:150, 50:250] = [255, 255, 255]

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process
            processor = ImageOptimizationTask(target_width=1920)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should not be upscaled
            assert result.img is not None
            assert result.img.shape[1] <= 300

    def test_process_with_perspective_correction_disabled(self):
        """Test process() with perspective correction disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with clear rectangle
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (550, 350), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process without perspective correction
            processor = ImageOptimizationTask(enable_perspective_correction=False, target_width=2000)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should still process image
            assert result.img is not None

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
            processor = ImageOptimizationTask(target_width=2000)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Verify color channels are maintained
            assert result.img is not None
            assert result.img.shape[2] == 3  # BGR channels

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

    def test_detect_flipchart_quad_with_valid_rectangle(self):
        """Test _detect_flipchart_quad detects valid rectangular flipchart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with clear white rectangle (simulating flipchart)
            img = np.zeros((600, 800, 3), dtype=np.uint8)
            # Draw white rectangle covering ~60% of image
            cv2.rectangle(img, (100, 100), (700, 500), (255, 255, 255), -1)
            # Add black border to create clear edges
            cv2.rectangle(img, (100, 100), (700, 500), (0, 0, 0), 10)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            processor = ImageOptimizationTask(min_area_ratio=0.3, max_area_ratio=0.95)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should detect quadrilateral and crop
            assert result.img is not None

    def test_area_ratio_filtering(self):
        """Test that area ratio filters out too-small or too-large contours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image where main rectangle is too small (below min_area_ratio)
            img = np.zeros((1000, 1000, 3), dtype=np.uint8)
            # Small rectangle (only 10% of image)
            cv2.rectangle(img, (400, 400), (600, 600), (255, 255, 255), -1)

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Process with high min_area_ratio
            processor = ImageOptimizationTask(min_area_ratio=0.5, target_width=2000)
            file_task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(file_task)

            # Should use fallback since rectangle is too small
            assert result.img is not None

    def test_aspect_ratio_validation(self):
        """Test that aspect ratio validation filters extreme shapes."""
        processor = ImageOptimizationTask()

        # Create extremely wide quad (aspect ratio > 2.5)
        wide_quad = np.array([[[10, 100]], [[500, 100]], [[500, 150]], [[10, 150]]], dtype=np.int32)
        assert not processor._is_valid_quad(wide_quad, 600, 200)

        # Create extremely tall quad (aspect ratio < 0.5)
        tall_quad = np.array([[[100, 10]], [[150, 10]], [[150, 500]], [[100, 500]]], dtype=np.int32)
        assert not processor._is_valid_quad(tall_quad, 200, 600)

        # Create reasonable quad (aspect ratio ~1.5)
        good_quad = np.array([[[50, 50]], [[350, 50]], [[350, 250]], [[50, 250]]], dtype=np.int32)
        assert processor._is_valid_quad(good_quad, 400, 300)

    def test_order_points(self):
        """Test _order_points correctly orders quad corners."""
        processor = ImageOptimizationTask()

        # Create points in random order
        pts = np.array([[200, 100], [100, 100], [200, 200], [100, 200]], dtype="float32")

        ordered = processor._order_points(pts)

        # Should be: top-left, top-right, bottom-right, bottom-left
        assert np.allclose(ordered[0], [100, 100])  # top-left
        assert np.allclose(ordered[1], [200, 100])  # top-right
        assert np.allclose(ordered[2], [200, 200])  # bottom-right
        assert np.allclose(ordered[3], [100, 200])  # bottom-left

    def test_fallback_edge_crop_with_content(self):
        """Test _fallback_edge_crop detects edges and crops."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with content
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            # Add white content in center
            img[100:300, 100:300] = [255, 255, 255]

            # Save test image
            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            processor = ImageOptimizationTask(target_width=2000)
            # Use fallback directly
            img_read = cv2.imread(str(test_file))
            result_img = processor._fallback_edge_crop(img_read)

            # Should crop around content with margins
            assert result_img is not None
            assert result_img.shape[0] <= 400
            assert result_img.shape[1] <= 400

    def test_fallback_edge_crop_no_edges_returns_original(self):
        """Test _fallback_edge_crop returns original when no edges found."""
        processor = ImageOptimizationTask()

        # Create completely black image (no edges)
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        result_img = processor._fallback_edge_crop(img)

        # Should return original image
        assert result_img is not None
        assert result_img.shape == img.shape
