from typing import Dict, Type, List

from .task_item import TaskProcessor


class TaskRegistry:
    """
    Registry for discovering and loading task processors.
    Allows tasks to register themselves and be looked up by name.
    """

    _registry: Dict[str, Type[TaskProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a task processor.

        Usage:
            @TaskRegistry.register("my_task")
            class MyTask(TaskProcessor):
                pass
        """

        def decorator(task_class: Type[TaskProcessor]):
            if name in cls._registry:
                raise ValueError(f"Task '{name}' is already registered")
            cls._registry[name] = task_class
            return task_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[TaskProcessor]:
        """
        Get task class by name.

        Args:
            name: The registered name of the task

        Returns:
            The task class

        Raises:
            KeyError: If task name is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Task '{name}' not found in registry. Available: {cls.list_tasks()}")
        return cls._registry[name]

    @classmethod
    def list_tasks(cls) -> List[str]:
        """
        List all registered task names.

        Returns:
            List of registered task names
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clear the registry. Useful for testing.
        """
        cls._registry.clear()
