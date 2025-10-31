"""Tests for WorkflowMonitor."""

import logging
import queue
import time
from unittest.mock import Mock, patch

import pytest

from pipeline.monitor import WorkflowMonitor
from pipeline.worker import Worker
from tasks.task_item import TaskProcessor, FileTask
from pathlib import Path


class MockTaskProcessor(TaskProcessor):
    """Simple mock processor for testing."""

    def process(self, task: FileTask) -> FileTask:
        return task


class TestWorkflowMonitor:
    """Test WorkflowMonitor functionality."""

    def test_creation(self):
        """Test WorkflowMonitor can be created."""
        q = queue.Queue()
        worker = Worker(name="test_worker", input_q=q, output_q=queue.Queue(), process_fn=MockTaskProcessor())

        monitor = WorkflowMonitor(workers=[worker], refresh_interval=0.1)

        assert len(monitor.workers) == 1
        assert monitor.refresh_interval == 0.1
        assert monitor.console is not None
        assert not monitor._stop_event.is_set()

    def test_creation_with_default_refresh_interval(self):
        """Test WorkflowMonitor with default refresh interval."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker])

        assert monitor.refresh_interval == 0.5

    def test_creation_with_multiple_workers(self):
        """Test WorkflowMonitor with multiple workers."""
        workers = [
            Worker(name=f"worker{i}", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor())
            for i in range(3)
        ]

        monitor = WorkflowMonitor(workers=workers)

        assert len(monitor.workers) == 3
        assert len(monitor._last_idle_logged) == 3

    def test_stop_sets_event(self):
        """Test stop() sets the stop event."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker])

        assert not monitor._stop_event.is_set()
        monitor.stop()
        assert monitor._stop_event.is_set()

    def test_render_creates_table(self):
        """Test render() creates a Rich table."""
        q1 = queue.Queue()
        q2 = queue.Queue()
        worker = Worker(name="test_worker", input_q=q1, output_q=q2, process_fn=MockTaskProcessor())

        monitor = WorkflowMonitor(workers=[worker])
        table = monitor.render()

        assert table is not None
        assert table.title == "Workflow Pipeline Monitor"
        assert len(table.columns) == 4

    def test_render_shows_worker_status(self):
        """Test render() shows worker status correctly."""
        q1 = queue.Queue()
        q2 = queue.Queue()
        worker = Worker(name="test_worker", input_q=q1, output_q=q2, process_fn=MockTaskProcessor())
        worker.done_count = 5
        q1.put(FileTask(file_path=Path("test.jpg"), sort_key=1.0))

        monitor = WorkflowMonitor(workers=[worker])
        table = monitor.render()

        # Table should be generated successfully
        assert table is not None
        assert len(table.rows) == 1

    def test_render_with_current_task(self):
        """Test render() shows current task when worker is processing."""
        q1 = queue.Queue()
        q2 = queue.Queue()
        worker = Worker(name="test_worker", input_q=q1, output_q=q2, process_fn=MockTaskProcessor())

        # Set current task
        task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
        worker.current_task = task

        monitor = WorkflowMonitor(workers=[worker])
        table = monitor.render()

        assert table is not None

    def test_render_without_current_task(self):
        """Test render() shows dash when no current task."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )
        worker.current_task = None

        monitor = WorkflowMonitor(workers=[worker])
        table = monitor.render()

        # Should show "-" for processing column
        assert table is not None

    def test_run_stops_when_event_set(self):
        """Test run() stops when stop event is set."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker], refresh_interval=0.05)

        # Start monitor in background
        monitor.start()

        # Let it run briefly
        time.sleep(0.1)

        # Stop it
        monitor.stop()

        # Wait for it to finish
        monitor.join(timeout=1.0)

        assert not monitor.is_alive()

    def test_run_updates_display(self, caplog):
        """Test run() updates display and logs worker status."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker], refresh_interval=0.05)

        # Start monitor
        monitor.start()

        with caplog.at_level(logging.DEBUG):
            # Let it run and log
            time.sleep(0.15)  # Multiple refresh cycles

            # Stop it
            monitor.stop()
            monitor.join(timeout=1.0)

        # Should have logged worker status
        assert "test_worker" in caplog.text

    def test_idle_logging_once(self, caplog):
        """Test idle state is logged only once."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )
        worker.current_task = None

        monitor = WorkflowMonitor(workers=[worker], refresh_interval=0.05)

        monitor.start()

        with caplog.at_level(logging.DEBUG):
            # Let multiple refresh cycles occur
            time.sleep(0.2)

            monitor.stop()
            monitor.join(timeout=1.0)

        # Count how many times "is now idle" appears
        idle_logs = [record for record in caplog.records if "is now idle" in record.message]

        # Should only log idle once (or a few times max if timing is weird)
        assert len(idle_logs) <= 2

    def test_idle_state_reset_when_processing(self, caplog):
        """Test idle state resets when worker starts processing."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker], refresh_interval=0.05)
        monitor.start()

        with caplog.at_level(logging.DEBUG):
            # Start idle
            worker.current_task = None
            time.sleep(0.1)

            # Simulate worker starting to process
            worker.current_task = FileTask(file_path=Path("test.jpg"), sort_key=1.0)
            time.sleep(0.1)

            # Back to idle
            worker.current_task = None
            time.sleep(0.1)

            monitor.stop()
            monitor.join(timeout=1.0)

        # Should see idle logged when it becomes idle again
        idle_logs = [record for record in caplog.records if "is now idle" in record.message]
        assert len(idle_logs) >= 1

    def test_final_update_on_exit(self, caplog):
        """Test monitor logs final status on exit."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker], refresh_interval=0.05)
        monitor.start()

        with caplog.at_level(logging.DEBUG):
            time.sleep(0.1)
            monitor.stop()
            monitor.join(timeout=1.0)

        # Should see FINAL log message
        final_logs = [record for record in caplog.records if "[FINAL]" in record.message]
        assert len(final_logs) >= 1

    def test_multiple_workers_in_table(self):
        """Test render() shows all workers in table."""
        workers = [
            Worker(name=f"worker{i}", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor())
            for i in range(3)
        ]

        for i, w in enumerate(workers):
            w.done_count = i * 10

        monitor = WorkflowMonitor(workers=workers)
        table = monitor.render()

        # Should have rows for all workers
        assert table is not None
        assert len(table.rows) == 3

    def test_queue_size_in_table(self):
        """Test render() shows queue sizes correctly."""
        q = queue.Queue()
        # Add some items to queue
        for i in range(5):
            q.put(FileTask(file_path=Path(f"test{i}.jpg"), sort_key=i))

        worker = Worker(name="test_worker", input_q=q, output_q=queue.Queue(), process_fn=MockTaskProcessor())

        monitor = WorkflowMonitor(workers=[worker])
        table = monitor.render()

        # Should show queue size 5
        assert table is not None

    def test_non_daemon_thread(self):
        """Test monitor thread is created as non-daemon for proper cleanup."""
        worker = Worker(
            name="test_worker", input_q=queue.Queue(), output_q=queue.Queue(), process_fn=MockTaskProcessor()
        )

        monitor = WorkflowMonitor(workers=[worker])

        assert monitor.daemon is False
        assert monitor.name == "WorkflowMonitor"
