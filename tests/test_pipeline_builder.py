"""
Tests for PipelineBuilder and configuration system.
"""

import pytest
from pathlib import Path

from pipeline.config import TaskConfig, PipelineConfig, load_config_from_dict
from pipeline.builder import PipelineBuilder
from tasks.task_item import TaskProcessor, FileTask
from tasks.registry import TaskRegistry


class SimpleTestTask(TaskProcessor):
    """Simple task for testing."""

    def __init__(self, value: int = 0):
        self.value = value

    def process(self, task: FileTask) -> FileTask:
        task.sort_key = float(self.value)
        return task


class TestPipelineConfig:
    """Test configuration dataclasses."""

    def test_task_config_creation(self):
        """Test creating a TaskConfig."""
        config = TaskConfig(name="test", task_type="test_task", params={"key": "value"})

        assert config.name == "test"
        assert config.task_type == "test_task"
        assert config.params == {"key": "value"}

    def test_task_config_default_params(self):
        """Test TaskConfig with default params."""
        config = TaskConfig(name="test", task_type="test_task")

        assert config.params == {}

    def test_pipeline_config_creation(self):
        """Test creating a PipelineConfig."""
        task_cfg = TaskConfig(name="task1", task_type="type1")
        config = PipelineConfig(tasks=[task_cfg], input_dir="images/", extension=".png")

        assert len(config.tasks) == 1
        assert config.input_dir == "images/"
        assert config.extension == ".png"

    def test_pipeline_config_requires_tasks(self):
        """Test that PipelineConfig requires at least one task."""
        with pytest.raises(ValueError) as exc_info:
            PipelineConfig(tasks=[])

        assert "at least one task" in str(exc_info.value)

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        data = {
            "tasks": [
                {"name": "task1", "task_type": "type1", "params": {"p1": 10}},
                {"name": "task2", "task_type": "type2"},
            ],
            "input_dir": "test/",
            "extension": ".jpg",
            "output_file": "output.pdf",
        }

        config = load_config_from_dict(data)

        assert len(config.tasks) == 2
        assert config.tasks[0].name == "task1"
        assert config.tasks[0].params == {"p1": 10}
        assert config.tasks[1].params == {}
        assert config.input_dir == "test/"
        assert config.extension == ".jpg"


class TestPipelineBuilder:
    """Test PipelineBuilder functionality."""

    def setup_method(self):
        """Register test tasks before each test."""
        TaskRegistry.clear()
        TaskRegistry.register("simple_test")(SimpleTestTask)

    def teardown_method(self):
        """Clean up registry after each test."""
        TaskRegistry.clear()

    def test_builder_creation(self):
        """Test creating a PipelineBuilder."""
        task_cfg = TaskConfig(name="test", task_type="simple_test")
        config = PipelineConfig(tasks=[task_cfg])

        builder = PipelineBuilder(config)

        assert builder.config == config
        assert len(builder.providers) == 0

    def test_register_provider(self):
        """Test registering a provider."""
        task_cfg = TaskConfig(name="test", task_type="simple_test")
        config = PipelineConfig(tasks=[task_cfg])
        builder = PipelineBuilder(config)

        mock_provider = "mock_provider_instance"
        builder.register_provider("test_provider", mock_provider)

        assert "test_provider" in builder.providers
        assert builder.providers["test_provider"] == mock_provider

    def test_build_creates_workers_and_queues(self):
        """Test that build creates workers and queues."""
        task_cfg = TaskConfig(name="test", task_type="simple_test")
        config = PipelineConfig(tasks=[task_cfg])
        builder = PipelineBuilder(config)

        workers, queues = builder.build()

        assert len(workers) == 1
        assert len(queues) == 2  # N+1 queues for N tasks
        assert workers[0].name == "test"

    def test_build_multiple_tasks(self):
        """Test building pipeline with multiple tasks."""
        config = PipelineConfig(
            tasks=[
                TaskConfig(name="task1", task_type="simple_test"),
                TaskConfig(name="task2", task_type="simple_test"),
                TaskConfig(name="task3", task_type="simple_test"),
            ]
        )
        builder = PipelineBuilder(config)

        workers, queues = builder.build()

        assert len(workers) == 3
        assert len(queues) == 4
        assert workers[0].name == "task1"
        assert workers[1].name == "task2"
        assert workers[2].name == "task3"

    def test_dependency_injection_with_provider(self):
        """Test dependency injection with registered providers."""

        # Register task that takes a parameter
        @TaskRegistry.register("param_task")
        class ParamTask(TaskProcessor):
            def __init__(self, provider):
                self.provider = provider

            def process(self, task: FileTask) -> FileTask:
                return task

        mock_provider = "injected_provider"
        config = PipelineConfig(
            tasks=[TaskConfig(name="test", task_type="param_task", params={"provider": "@test_provider"})]
        )

        builder = PipelineBuilder(config)
        builder.register_provider("test_provider", mock_provider)

        workers, queues = builder.build()

        # Check that provider was injected
        assert workers[0].process_fn.provider == mock_provider

    def test_dependency_injection_missing_provider_raises_error(self):
        """Test that missing provider raises error."""

        @TaskRegistry.register("needs_provider")
        class NeedsProviderTask(TaskProcessor):
            def __init__(self, provider):
                self.provider = provider

            def process(self, task: FileTask) -> FileTask:
                return task

        config = PipelineConfig(
            tasks=[TaskConfig(name="test", task_type="needs_provider", params={"provider": "@missing_provider"})]
        )

        builder = PipelineBuilder(config)

        with pytest.raises(KeyError) as exc_info:
            builder.build()

        assert "missing_provider" in str(exc_info.value)
        assert "not registered" in str(exc_info.value)

    def test_task_instantiation_with_params(self):
        """Test that task parameters are passed correctly."""
        config = PipelineConfig(tasks=[TaskConfig(name="test", task_type="simple_test", params={"value": 42})])

        builder = PipelineBuilder(config)
        workers, queues = builder.build()

        # Check that parameter was passed
        assert workers[0].process_fn.value == 42

    def test_nonexistent_task_type_raises_error(self):
        """Test that nonexistent task type raises error."""
        config = PipelineConfig(tasks=[TaskConfig(name="test", task_type="nonexistent_task")])

        builder = PipelineBuilder(config)

        with pytest.raises((KeyError, ImportError)):
            builder.build()
