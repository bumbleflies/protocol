"""
Tests for TaskRegistry (OCP compliance).
"""
import pytest

from tasks.registry import TaskRegistry
from tasks.task_item import TaskProcessor, FileTask
from pathlib import Path


class DummyTask(TaskProcessor):
    """Dummy task for testing."""

    def process(self, task: FileTask) -> FileTask:
        return task


class TestTaskRegistry:
    """Test TaskRegistry functionality and OCP compliance."""

    def setup_method(self):
        """Clear registry before each test."""
        TaskRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        TaskRegistry.clear()

    def test_register_task(self):
        """Test registering a task in the registry."""
        @TaskRegistry.register("test_task")
        class TestTask(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                return task

        assert "test_task" in TaskRegistry.list_tasks()

    def test_get_registered_task(self):
        """Test retrieving a registered task."""
        @TaskRegistry.register("dummy")
        class DummyTestTask(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                return task

        task_class = TaskRegistry.get("dummy")
        assert task_class == DummyTestTask

    def test_get_nonexistent_task_raises_error(self):
        """Test that getting a nonexistent task raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            TaskRegistry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "not found in registry" in str(exc_info.value)

    def test_register_duplicate_name_raises_error(self):
        """Test that registering the same name twice raises error."""
        @TaskRegistry.register("duplicate")
        class Task1(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                return task

        with pytest.raises(ValueError) as exc_info:
            @TaskRegistry.register("duplicate")
            class Task2(TaskProcessor):
                def process(self, task: FileTask) -> FileTask:
                    return task

        assert "already registered" in str(exc_info.value)

    def test_list_tasks(self):
        """Test listing all registered tasks."""
        @TaskRegistry.register("task1")
        class Task1(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                return task

        @TaskRegistry.register("task2")
        class Task2(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                return task

        tasks = TaskRegistry.list_tasks()
        assert len(tasks) == 2
        assert "task1" in tasks
        assert "task2" in tasks

    def test_clear_registry(self):
        """Test clearing the registry."""
        @TaskRegistry.register("temp_task")
        class TempTask(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                return task

        assert len(TaskRegistry.list_tasks()) > 0

        TaskRegistry.clear()

        assert len(TaskRegistry.list_tasks()) == 0

    def test_ocp_compliance_new_tasks_without_modification(self):
        """
        Test OCP: New tasks can be added without modifying existing code.
        This demonstrates Open/Closed Principle compliance.
        """
        # Register first task
        @TaskRegistry.register("existing_task")
        class ExistingTask(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                task.sort_key = 1.0
                return task

        # Add new task without modifying ExistingTask
        @TaskRegistry.register("new_task")
        class NewTask(TaskProcessor):
            def process(self, task: FileTask) -> FileTask:
                task.sort_key = 2.0
                return task

        # Both tasks available
        existing = TaskRegistry.get("existing_task")
        new = TaskRegistry.get("new_task")

        # Both work independently
        test_task = FileTask(file_path=Path("test.jpg"), sort_key=0.0)

        result1 = existing().process(test_task)
        assert result1.sort_key == 1.0

        test_task.sort_key = 0.0  # Reset
        result2 = new().process(test_task)
        assert result2.sort_key == 2.0
