from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskConfig:
    """
    Configuration for a single task in the pipeline.
    """

    name: str
    task_type: str  # Registry name or fully qualified class name
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    Defines the sequence of tasks and their parameters.
    """

    tasks: List[TaskConfig]
    input_dir: str = "."
    extension: str = ".jpg"
    output_file: str = "combined.pdf"
    enable_ocr: bool = True

    def __post_init__(self):
        if not self.tasks:
            raise ValueError("Pipeline must have at least one task")


def load_config_from_dict(data: Dict[str, Any]) -> PipelineConfig:
    """
    Load PipelineConfig from a dictionary (e.g., from YAML/JSON).

    Args:
        data: Dictionary with pipeline configuration

    Returns:
        PipelineConfig instance

    Example:
        data = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {}}
            ],
            "input_dir": "images/",
            "extension": ".jpg"
        }
        config = load_config_from_dict(data)
    """
    tasks = [
        TaskConfig(name=t["name"], task_type=t["task_type"], params=t.get("params", {})) for t in data.get("tasks", [])
    ]

    return PipelineConfig(
        tasks=tasks,
        input_dir=data.get("input_dir", "."),
        extension=data.get("extension", ".jpg"),
        output_file=data.get("output_file", "combined.pdf"),
        enable_ocr=data.get("enable_ocr", True),
    )
