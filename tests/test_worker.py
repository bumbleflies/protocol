"""
Tests for Worker and its interaction with TaskProcessor.
"""

import pytest
import queue
import time
from pathlib import Path

from pipeline.worker import Worker
from tasks.task_item import TaskProcessor, FinalizableTaskProcessor, FileTask, FinalizeTask


class CountingProcessor(TaskProcessor):
    """Processor that counts how many times it was called."""

    def __init__(self):
        self.count = 0

    def process(self, task: FileTask) -> FileTask:
        self.count += 1
        task.sort_key = float(self.count)
        return task


class FinalizableCountingProcessor(FinalizableTaskProcessor):
    """Finalizable processor for testing."""

    def __init__(self):
        self.process_count = 0
        self.finalize_count = 0

    def process(self, task: FileTask) -> FileTask:
        self.process_count += 1
        return task

    def finalize(self) -> None:
        self.finalize_count += 1


class TestWorkerWithTaskProcessor:
    """Test Worker integration with TaskProcessor."""

    def test_worker_processes_tasks(self):
        """Test that worker processes FileTask correctly."""
        input_q = queue.Queue()
        output_q = queue.Queue()
        processor = CountingProcessor()

        worker = Worker("test", input_q, output_q, processor)
        worker.start()

        # Send tasks
        task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0)
        task2 = FileTask(file_path=Path("test2.jpg"), sort_key=2.0)

        input_q.put(task1)
        input_q.put(task2)
        input_q.put(FinalizeTask())

        # Wait for processing
        time.sleep(0.5)

        worker.stop()
        worker.join(timeout=2)

        assert processor.count == 2

        # Check output queue
        result1 = output_q.get()
        assert result1.sort_key == 1.0

        result2 = output_q.get()
        assert result2.sort_key == 2.0

    def test_worker_handles_finalize_task(self):
        """Test that worker correctly handles FinalizeTask."""
        input_q = queue.Queue()
        output_q = queue.Queue()
        processor = FinalizableCountingProcessor()

        worker = Worker("test", input_q, output_q, processor)
        worker.start()

        # Send tasks and finalize
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
        input_q.put(task)
        input_q.put(FinalizeTask())

        # Wait for processing
        time.sleep(0.5)

        worker.stop()
        worker.join(timeout=2)

        assert processor.process_count == 1
        assert processor.finalize_count == 1

    def test_worker_with_non_finalizable_processor(self):
        """Test worker with regular TaskProcessor (not finalizable)."""
        input_q = queue.Queue()
        output_q = queue.Queue()
        processor = CountingProcessor()  # Not finalizable

        worker = Worker("test", input_q, output_q, processor)
        worker.start()

        # Send finalize task
        input_q.put(FinalizeTask())

        # Wait for processing
        time.sleep(0.5)

        worker.stop(timeout=2)
        worker.join(timeout=2)

        # Should not crash, just pass through
        assert worker.finalize_done_event.is_set()

    def test_worker_stop_with_timeout(self):
        """Test that worker stop respects timeout."""
        input_q = queue.Queue()
        processor = CountingProcessor()

        worker = Worker("test", input_q, None, processor)
        worker.start()

        # Don't send FinalizeTask, so stop should timeout
        start = time.time()
        worker.stop(timeout=0.5)
        elapsed = time.time() - start

        # Should timeout after ~0.5 seconds
        assert 0.4 < elapsed < 1.0

        # Force stop
        worker._stop_event.set()
        worker.join(timeout=1)

    def test_worker_done_count(self):
        """Test that worker tracks done count correctly."""
        input_q = queue.Queue()
        output_q = queue.Queue()
        processor = CountingProcessor()

        worker = Worker("test", input_q, output_q, processor)
        worker.start()

        # Send multiple tasks
        for i in range(5):
            task = FileTask(file_path=Path(f"test{i}.jpg"), sort_key=float(i))
            input_q.put(task)

        input_q.put(FinalizeTask())

        # Wait for processing
        time.sleep(0.5)

        assert worker.done_count == 6  # 5 tasks + 1 finalize

        worker.stop()
        worker.join(timeout=2)

    def test_worker_error_handling(self):
        """Test that worker handles processor errors gracefully."""

        class FailingProcessor(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                raise RuntimeError("Test error")

        input_q = queue.Queue()
        output_q = queue.Queue()
        processor = FailingProcessor()

        worker = Worker("test", input_q, output_q, processor)
        worker.start()

        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
        input_q.put(task)
        input_q.put(FinalizeTask())

        # Wait for processing
        time.sleep(0.5)

        # Worker should not crash, task should pass through
        result = output_q.get(timeout=1)
        assert result == task  # Original task passed through

        worker.stop()
        worker.join(timeout=2)
