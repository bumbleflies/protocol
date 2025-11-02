"""Tests for PDFSaveTask."""

import logging
from pathlib import Path
import tempfile

import numpy as np
import pytest

from flipchart_ocr_pipeline.tasks.save_pdf import PDFSaveTask
from flipchart_ocr_pipeline.tasks.task_item import FileTask, OCRBox


class TestPDFSaveTask:
    """Test PDFSaveTask processor."""

    def test_creation(self):
        """Test PDFSaveTask can be created."""
        task = PDFSaveTask(output_path="test.pdf")
        assert task.output_path == Path("test.pdf")
        assert task.collected_tasks == []

    def test_process_collects_tasks(self):
        """Test process() collects FileTask with image."""
        processor = PDFSaveTask(output_path="output.pdf")

        # Create a FileTask with an image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

        result = processor.process(file_task)

        assert result == file_task
        assert len(processor.collected_tasks) == 1
        assert processor.collected_tasks[0] == file_task

    def test_process_skips_tasks_without_image(self):
        """Test process() skips FileTask without image."""
        processor = PDFSaveTask(output_path="output.pdf")

        # Create a FileTask without an image
        file_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)

        result = processor.process(file_task)

        assert result == file_task
        assert len(processor.collected_tasks) == 0

    def test_process_raises_type_error_for_invalid_task(self):
        """Test process() raises TypeError for invalid task type."""
        processor = PDFSaveTask(output_path="output.pdf")

        with pytest.raises(TypeError, match="Expected FileTask or StatusTask"):
            processor.process("invalid_task")

    def test_finalize_with_no_tasks(self, caplog):
        """Test finalize() with no collected tasks."""
        processor = PDFSaveTask(output_path="output.pdf")

        with caplog.at_level(logging.WARNING):
            processor.finalize()

        assert "No images to save in PDF" in caplog.text

    def test_finalize_creates_pdf(self):
        """Test finalize() creates PDF from collected tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test images
            img1 = np.zeros((100, 100, 3), dtype=np.uint8)
            img1[:, :] = [255, 0, 0]  # Blue in BGR

            img2 = np.zeros((100, 100, 3), dtype=np.uint8)
            img2[:, :] = [0, 255, 0]  # Green in BGR

            # Create FileTasks
            task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0, img=img1)
            task2 = FileTask(file_path=Path("test2.jpg"), sort_key=2.0, img=img2)

            processor.process(task1)
            processor.process(task2)

            # Finalize to create PDF
            processor.finalize()

            # Verify PDF was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_finalize_sorts_by_sort_key(self):
        """Test finalize() sorts tasks by sort_key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test images with different colors
            img1 = np.zeros((50, 50, 3), dtype=np.uint8)
            img2 = np.zeros((50, 50, 3), dtype=np.uint8)
            img3 = np.zeros((50, 50, 3), dtype=np.uint8)

            # Add tasks in non-sorted order
            task3 = FileTask(file_path=Path("test3.jpg"), sort_key=3.0, img=img3)
            task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0, img=img1)
            task2 = FileTask(file_path=Path("test2.jpg"), sort_key=2.0, img=img2)

            processor.process(task3)
            processor.process(task1)
            processor.process(task2)

            # Finalize should sort by sort_key
            processor.finalize()

            assert output_path.exists()

    def test_finalize_with_ocr_annotations(self):
        """Test finalize() adds OCR annotations to PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test image
            img = np.zeros((200, 200, 3), dtype=np.uint8)

            # Create OCR boxes
            ocr_boxes = [
                OCRBox(label="Test Text", confidence=0.9, x1=10, y1=10, x2=100, y2=10, x3=100, y3=50, x4=10, y4=50)
            ]

            task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, ocr_boxes=ocr_boxes)

            processor.process(task)
            processor.finalize()

            # Verify PDF was created
            assert output_path.exists()

    def test_finalize_skips_low_confidence_annotations(self, caplog):
        """Test finalize() skips OCR boxes with low confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test image
            img = np.zeros((200, 200, 3), dtype=np.uint8)

            # Create OCR box with low confidence
            ocr_boxes = [
                OCRBox(
                    label="Low Confidence",
                    confidence=0.3,  # Below 0.5 threshold
                    x1=10,
                    y1=10,
                    x2=100,
                    y2=10,
                    x3=100,
                    y3=50,
                    x4=10,
                    y4=50,
                )
            ]

            task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, ocr_boxes=ocr_boxes)

            processor.process(task)

            with caplog.at_level(logging.DEBUG):
                processor.finalize()

            assert "Skipping low confidence annotation" in caplog.text

    def test_finalize_handles_annotation_errors(self, caplog):
        """Test finalize() handles annotation creation errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test image
            img = np.zeros((200, 200, 3), dtype=np.uint8)

            # Create invalid OCR box that might cause annotation error
            ocr_boxes = [
                OCRBox(
                    label="Test",
                    confidence=0.9,
                    x1=-1000,
                    y1=-1000,  # Invalid coordinates
                    x2=-900,
                    y2=-1000,
                    x3=-900,
                    y3=-950,
                    x4=-1000,
                    y4=-950,
                )
            ]

            task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, ocr_boxes=ocr_boxes)

            processor.process(task)
            processor.finalize()

            # Should still create PDF despite annotation error
            assert output_path.exists()

    def test_finalize_removes_temp_file(self):
        """Test finalize() removes temporary PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

            processor.process(task)
            processor.finalize()

            # Verify temp file was removed
            temp_pdf_path = output_path.with_suffix(".temp.pdf")
            assert not temp_pdf_path.exists()

    def test_finalize_with_empty_label(self):
        """Test finalize() handles OCR boxes with empty labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create test image
            img = np.zeros((200, 200, 3), dtype=np.uint8)

            # Create OCR box with empty label
            ocr_boxes = [
                OCRBox(
                    label="", confidence=0.9, x1=10, y1=10, x2=100, y2=10, x3=100, y3=50, x4=10, y4=50  # Empty label
                )
            ]

            task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img, ocr_boxes=ocr_boxes)

            processor.process(task)
            processor.finalize()

            # Should create PDF with empty annotation text
            assert output_path.exists()

    def test_multiple_pages_with_different_ocr_boxes(self):
        """Test finalize() handles multiple pages with different OCR boxes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create two test images with different OCR boxes
            img1 = np.zeros((200, 200, 3), dtype=np.uint8)
            img2 = np.zeros((200, 200, 3), dtype=np.uint8)

            ocr_boxes1 = [
                OCRBox(label="Page 1", confidence=0.9, x1=10, y1=10, x2=100, y2=10, x3=100, y3=50, x4=10, y4=50)
            ]

            ocr_boxes2 = [
                OCRBox(label="Page 2", confidence=0.85, x1=50, y1=50, x2=150, y2=50, x3=150, y3=90, x4=50, y4=90)
            ]

            task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0, img=img1, ocr_boxes=ocr_boxes1)

            task2 = FileTask(file_path=Path("test2.jpg"), sort_key=2.0, img=img2, ocr_boxes=ocr_boxes2)

            processor.process(task1)
            processor.process(task2)
            processor.finalize()

            # Verify multi-page PDF was created
            assert output_path.exists()

    def test_add_annotations_with_more_pages_than_tasks(self):
        """Test _add_annotations() handles PDFs with more pages than tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"
            processor = PDFSaveTask(output_path=str(output_path))

            # Create a single test image with OCR boxes
            img1 = np.zeros((200, 200, 3), dtype=np.uint8)
            ocr_boxes1 = [
                OCRBox(label="Page 1", confidence=0.9, x1=10, y1=10, x2=100, y2=10, x3=100, y3=50, x4=10, y4=50)
            ]
            task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0, img=img1, ocr_boxes=ocr_boxes1)

            # Process and create initial PDF
            processor.process(task1)
            processor.finalize()

            # Now manually create a PDF with more pages than tasks
            # Read the created PDF and add an extra blank page
            from pypdf import PdfReader, PdfWriter

            reader = PdfReader(str(output_path))
            writer = PdfWriter()

            # Add original page
            writer.add_page(reader.pages[0])

            # Add an extra blank page (more pages than tasks)
            writer.add_blank_page(width=200, height=200)

            # Save modified PDF with extra page
            multi_page_path = Path(tmpdir) / "multi_page.pdf"
            with open(multi_page_path, "wb") as f:
                writer.write(f)

            # Create a new processor and test _add_annotations with mismatched page count
            final_output = Path(tmpdir) / "final_output.pdf"
            processor2 = PDFSaveTask(output_path=str(final_output))

            # Call _add_annotations with 1 task but 2-page PDF
            # This triggers the edge case: if page_idx >= len(tasks): continue
            processor2._add_annotations(multi_page_path, final_output, [task1])

            # Verify final PDF was created successfully
            assert final_output.exists()
            assert final_output.stat().st_size > 0

            # Verify it has 2 pages (original + blank)
            final_reader = PdfReader(str(final_output))
            assert len(final_reader.pages) == 2
