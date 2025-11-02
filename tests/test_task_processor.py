"""
Tests for TaskProcessor base classes and LSP compliance.
"""

import numpy as np
from pathlib import Path

from flipchart_ocr_pipeline.tasks.task_item import TaskProcessor, FinalizableTaskProcessor, FileTask, OCRBox


class SimpleTaskProcessor(TaskProcessor):
    """Simple test processor that adds a marker to the task."""

    def process(self, task: FileTask) -> FileTask:
        task.sort_key = 999.0  # Marker
        return task


class FinalizableTestProcessor(FinalizableTaskProcessor):
    """Test processor that tracks finalization."""

    def __init__(self):
        self.finalize_called = False
        self.processed_count = 0

    def process(self, task: FileTask) -> FileTask:
        self.processed_count += 1
        return task

    def finalize(self) -> None:
        self.finalize_called = True


class TestTaskProcessor:
    """Test TaskProcessor base class and LSP compliance."""

    def test_simple_processor_implements_interface(self):
        """Test that TaskProcessor subclasses can be instantiated."""
        processor = SimpleTaskProcessor()
        assert isinstance(processor, TaskProcessor)

    def test_processor_processes_file_task(self):
        """Test that processor correctly processes FileTask."""
        processor = SimpleTaskProcessor()
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)

        result = processor.process(task)

        assert result.sort_key == 999.0
        assert result.file_path == Path("test.jpg")

    def test_finalizable_processor_implements_interface(self):
        """Test that FinalizableTaskProcessor subclasses work."""
        processor = FinalizableTestProcessor()
        assert isinstance(processor, FinalizableTaskProcessor)
        assert isinstance(processor, TaskProcessor)

    def test_finalizable_processor_can_process_and_finalize(self):
        """Test that FinalizableTaskProcessor can both process and finalize."""
        processor = FinalizableTestProcessor()

        # Process some tasks
        task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0)
        task2 = FileTask(file_path=Path("test2.jpg"), sort_key=2.0)

        processor.process(task1)
        processor.process(task2)

        assert processor.processed_count == 2
        assert not processor.finalize_called

        # Finalize
        processor.finalize()

        assert processor.finalize_called

    def test_lsp_compliance_substitutability(self):
        """
        Test LSP: FinalizableTaskProcessor can be used wherever TaskProcessor is expected.
        This tests the Liskov Substitution Principle.
        """

        def process_with_processor(processor: TaskProcessor, task: FileTask) -> FileTask:
            """Function that expects TaskProcessor."""
            return processor.process(task)

        # Should work with simple processor
        simple = SimpleTaskProcessor()
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
        result1 = process_with_processor(simple, task)
        assert result1 is not None

        # Should also work with finalizable processor (LSP)
        finalizable = FinalizableTestProcessor()
        result2 = process_with_processor(finalizable, task)
        assert result2 is not None


class TestFileTask:
    """Test FileTask dataclass."""

    def test_file_task_creation(self):
        """Test creating a FileTask."""
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.5)

        assert task.file_path == Path("test.jpg")
        assert task.sort_key == 1.5
        assert task.ocr_boxes == []
        assert task.asset_id is None
        assert task.img is None

    def test_file_task_with_ocr_boxes(self):
        """Test FileTask with OCR boxes."""
        box = OCRBox(label="Test", x1=0, y1=0, x2=10, y2=0, x3=10, y3=10, x4=0, y4=10, confidence=0.9)

        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, ocr_boxes=[box])

        assert len(task.ocr_boxes) == 1
        assert task.ocr_boxes[0].label == "Test"
        assert task.ocr_boxes[0].confidence == 0.9

    def test_file_task_with_image(self):
        """Test FileTask with image array."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0, img=img)

        assert task.img is not None
        assert task.img.shape == (100, 100, 3)
