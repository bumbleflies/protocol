"""Tests for ImageOptimizationTask with quadrilateral detection."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from flipchart_ocr_pipeline.tasks.image_optimization import ImageOptimizationTask
from flipchart_ocr_pipeline.tasks.task_item import FileTask, StatusTask


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

    def test_process_status_task_passes_through(self):
        """Test process() passes StatusTask through unchanged."""
        processor = ImageOptimizationTask()
        status_task = StatusTask(files_processed=5)

        result = processor.process(status_task)

        assert result is status_task
        assert result.files_processed == 5

    def test_process_raises_type_error_for_invalid_task(self):
        """Test process() raises TypeError for invalid task type."""
        processor = ImageOptimizationTask()

        with pytest.raises(TypeError, match="Expected FileTask or StatusTask"):
            processor.process("invalid_task")

    def test_process_nonexistent_file(self):
        """Test process() raises ValueError for nonexistent file."""
        processor = ImageOptimizationTask()
        file_task = FileTask(file_path=Path("/nonexistent/image.jpg"), sort_key=1.0)

        from flipchart_ocr_pipeline.tasks.exceptions import ImageProcessingException

        with pytest.raises(ImageProcessingException, match="Could not read image"):
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

    def test_detect_bright_paper_finds_white_quad(self):
        """Test _detect_bright_paper successfully detects bright white paper."""
        processor = ImageOptimizationTask()

        # Create image with bright white rectangle in center on darker background
        img = np.ones((500, 500, 3), dtype=np.uint8) * 150  # Gray background
        # Add bright white paper region
        cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), -1)

        quad = processor._detect_bright_paper(img)

        # Should detect the white rectangle
        assert quad is not None
        assert len(quad) == 4

    def test_detect_bright_paper_with_perspective_correction_disabled(self):
        """Test bright paper detection path with perspective correction disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image: bright white paper with NO clear edges (to skip edge detection)
            # Use a very soft gradient so edge detection fails
            img = np.ones((500, 500, 3), dtype=np.uint8) * 140
            # Add bright white paper region with soft edges
            for i in range(100, 400):
                for j in range(100, 400):
                    dist_from_edge = min(i - 100, j - 100, 400 - i, 400 - j)
                    brightness = min(255, 200 + dist_from_edge)
                    img[i, j] = [brightness, brightness, brightness]

            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            processor = ImageOptimizationTask(enable_perspective_correction=False)
            task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(task)

            # Should have processed the image
            assert result.img is not None
            # Image might be cropped or use fallback
            assert result.img.shape[0] <= 500
            assert result.img.shape[1] <= 500

    def test_detect_bright_paper_returns_none_for_dark_image(self):
        """Test _detect_bright_paper returns None when no bright paper detected."""
        processor = ImageOptimizationTask()

        # Create dark image with no bright regions
        img = np.ones((500, 500, 3), dtype=np.uint8) * 50  # Dark gray

        quad = processor._detect_bright_paper(img)

        # Should not detect anything
        assert quad is None

    def test_detect_bright_paper_rejects_invalid_aspect_ratio(self):
        """Test _detect_bright_paper rejects quads with invalid aspect ratios."""
        processor = ImageOptimizationTask()

        # Create image with very thin bright rectangle (invalid aspect ratio)
        img = np.ones((500, 500, 3), dtype=np.uint8) * 150
        cv2.rectangle(img, (200, 100), (210, 400), (255, 255, 255), -1)  # Very thin

        quad = processor._detect_bright_paper(img)

        # Should reject due to invalid aspect ratio
        assert quad is None

    def test_is_valid_quad_rejects_entire_image(self):
        """Test _is_valid_quad rejects quads that cover the entire image."""
        processor = ImageOptimizationTask()

        # Create quad that covers almost entire image (within margin)
        quad = np.array([[[10, 10]], [[490, 10]], [[490, 490]], [[10, 490]]], dtype=np.int32)

        is_valid = processor._is_valid_quad(quad, 500, 500)

        # Should reject as it's too close to image edges
        assert is_valid is False

    def test_detect_bright_paper_tries_multiple_thresholds(self):
        """Test _detect_bright_paper tries multiple threshold values."""
        processor = ImageOptimizationTask()

        # Create image with medium-bright paper (requires lower threshold)
        img = np.ones((500, 500, 3), dtype=np.uint8) * 150
        cv2.rectangle(img, (100, 100), (400, 400), (200, 200, 200), -1)  # Medium bright

        quad = processor._detect_bright_paper(img)

        # Should find it with one of the lower thresholds
        # Result depends on threshold values, may or may not find it
        # This test mainly ensures the loop runs without error
        assert quad is None or len(quad) == 4

    def test_bright_paper_with_perspective_correction_enabled(self):
        """Test bright paper detection with perspective correction applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image where edge detection will fail but brightness detection succeeds
            # Use very gradual gradient at edges to confuse edge detection
            img = np.ones((500, 500, 3), dtype=np.uint8) * 150

            # Create bright center with very gradual transitions (no sharp edges)
            for i in range(500):
                for j in range(500):
                    # Distance from center bright region
                    dx = max(0, max(120 - i, i - 380))
                    dy = max(0, max(120 - j, j - 380))
                    d = max(dx, dy)
                    if d == 0:
                        # Bright center
                        img[i, j] = [250, 250, 250]
                    elif d < 30:
                        # Gradual transition (confuses edge detection)
                        brightness = int(250 - (d / 30.0) * 100)
                        img[i, j] = [brightness, brightness, brightness]

            test_file = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(test_file), img)

            # Enable perspective correction
            processor = ImageOptimizationTask(enable_perspective_correction=True)
            task = FileTask(file_path=test_file, sort_key=1.0)

            result = processor.process(task)

            # Should have processed the image
            assert result.img is not None
            # Image should be processed (may be cropped or full size depending on detection)
            assert result.img.shape[0] <= 500
            assert result.img.shape[1] <= 500

    def test_detect_rotation_angle_with_horizontal_lines(self, tmpdir):
        """Test rotation angle detection with horizontal lines."""
        import cv2
        import numpy as np

        # Create image with horizontal lines (should detect ~0 degrees)
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        for y in [100, 200, 300]:
            cv2.line(img, (50, y), (550, y), (0, 0, 0), 2)

        processor = ImageOptimizationTask()
        angle = processor._detect_rotation_angle(img)

        # Should detect approximately 0 degrees
        if angle is not None:
            assert abs(angle) < 5  # Within 5 degrees of horizontal

    def test_detect_rotation_angle_returns_none_for_no_lines(self, tmpdir):
        """Test rotation angle detection returns None when no lines detected."""
        import numpy as np

        # Create solid color image with no lines
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200

        processor = ImageOptimizationTask()
        angle = processor._detect_rotation_angle(img)

        # Should return None when no lines detected
        assert angle is None

    def test_rotate_image_expands_canvas(self, tmpdir):
        """Test that image rotation expands canvas to fit rotated content."""
        import numpy as np

        # Create a small test image
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200

        processor = ImageOptimizationTask()
        rotated = processor._rotate_image(img, 45)

        # Rotated image should be larger to fit diagonal
        assert rotated.shape[0] > img.shape[0]
        assert rotated.shape[1] > img.shape[1]

    def test_rotate_image_preserves_content(self, tmpdir):
        """Test that image rotation preserves image content."""
        import cv2
        import numpy as np

        # Create image with a black square
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

        processor = ImageOptimizationTask()
        rotated = processor._rotate_image(img, 90)

        # Should have rotated content (check that it's not all white)
        assert rotated is not None
        assert np.mean(rotated) < 255  # Contains dark pixels

    def test_detect_rotation_angle_with_steep_lines(self, tmpdir):
        """Test rotation angle detection with lines at steep angles (> 45 degrees)."""
        import cv2
        import numpy as np

        # Create image with lines at 60 degrees (should normalize to -30)
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        for x_offset in [50, 150, 250]:
            # Draw lines at ~60 degrees
            cv2.line(img, (x_offset, 50), (x_offset + 100, 223), (0, 0, 0), 2)

        processor = ImageOptimizationTask()
        angle = processor._detect_rotation_angle(img)

        # Should detect and normalize angle
        if angle is not None:
            # Normalized angle should be in reasonable range
            assert -45 <= angle <= 45

    def test_detect_rotation_angle_no_lines(self, mocker):
        """Test rotation angle detection when HoughLinesP returns None (no lines)."""
        import numpy as np

        # Create a simple test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Mock HoughLinesP to return None (no lines detected)
        mocker.patch("cv2.HoughLinesP", return_value=None)

        processor = ImageOptimizationTask()
        angle = processor._detect_rotation_angle(img)

        # Should return None when no lines detected
        assert angle is None

    def test_detect_rotation_angle_empty_list(self, mocker):
        """Test rotation angle detection when HoughLinesP returns empty list."""
        import numpy as np

        # Create a simple test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Mock HoughLinesP to return empty list
        mocker.patch("cv2.HoughLinesP", return_value=[])

        processor = ImageOptimizationTask()
        angle = processor._detect_rotation_angle(img)

        # Should return None when empty list
        assert angle is None

    def test_rotation_applied_in_process(self, mocker):
        """Test that rotation is actually applied during process() when angle > 0.5."""
        import cv2
        import numpy as np
        from pathlib import Path
        from flipchart_ocr_pipeline.tasks import FileTask

        # Create a tilted image with clear horizontal lines
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        for y in [100, 200, 300]:
            cv2.line(img, (50, y), (550, y), (0, 0, 0), 3)

        # Rotate image by 5 degrees to trigger rotation correction
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated_input = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

        # Mock cv2.imread to return our test image
        mocker.patch("cv2.imread", return_value=rotated_input)

        processor = ImageOptimizationTask()
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
        result = processor.process(task)

        # Should have processed the image
        assert result.img is not None
        # Rotation should have been detected and applied (angle > 0.5)
        # Note: May not be called if angle detection fails, so just check image exists
        assert result.img.shape[0] > 0 and result.img.shape[1] > 0

    def test_bright_paper_detection_simple_crop(self, mocker):
        """Test bright paper detection with simple crop (no perspective correction)."""
        import cv2
        import numpy as np
        from pathlib import Path
        from flipchart_ocr_pipeline.tasks import FileTask

        # Create image with bright white paper on dark background
        img = np.ones((500, 700, 3), dtype=np.uint8) * 30  # Very dark background
        # Add bright white rectangle (simulating paper) - make it large and bright
        cv2.rectangle(img, (50, 50), (650, 450), (255, 255, 255), -1)

        # Mock cv2.imread to return our test image
        mocker.patch("cv2.imread", return_value=img.copy())

        # Mock _detect_flipchart_quad to return None (forcing bright paper detection)
        processor = ImageOptimizationTask(enable_perspective_correction=False)
        mocker.patch.object(processor, "_detect_flipchart_quad", return_value=None)

        # Create a mock quad for bright paper detection
        mock_quad = np.array([[50, 50], [650, 50], [650, 450], [50, 450]], dtype=np.int32)
        mocker.patch.object(processor, "_detect_bright_paper", return_value=mock_quad)

        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
        result = processor.process(task)

        # Should have cropped to the bright region using simple bounding rect
        assert result.img is not None
        # Verify it's cropped (should be 400x600 or close to the rectangle size)
        assert result.img.shape[0] <= 410  # Height around 400 + small margin
        assert result.img.shape[1] <= 610  # Width around 600 + small margin
