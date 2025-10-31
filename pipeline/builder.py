import importlib
import logging
import queue
from typing import Dict, Any, List, Tuple

from tasks.registry import TaskRegistry
from tasks.task_item import TaskProcessor
from .config import PipelineConfig
from .worker import Worker

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Builds a pipeline from configuration with dependency injection support.
    Allows registering providers and services that can be injected into tasks.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline builder.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.providers: Dict[str, Any] = {}

    def register_provider(self, name: str, provider: Any) -> None:
        """
        Register a provider for dependency injection.
        Providers can be referenced in task parameters using "@provider_name" syntax.

        Args:
            name: Provider name (e.g., "nvidia_uploader")
            provider: Provider instance

        Example:
            builder.register_provider("nvidia_uploader", NvidiaAssetUploader(api_key))
            # Then in config: params: {"uploader": "@nvidia_uploader"}
        """
        if name in self.providers:
            logger.warning(f"Provider '{name}' already registered, overwriting")
        self.providers[name] = provider
        logger.debug(f"Registered provider: {name}")

    def build(self) -> Tuple[List[Worker], List[queue.Queue]]:
        """
        Build workers and queues from configuration.

        Returns:
            Tuple of (workers list, queues list)

        Raises:
            KeyError: If a task type is not found in registry
            ValueError: If configuration is invalid
        """
        # Create queues: one per task + one final output queue
        queues = [queue.Queue() for _ in range(len(self.config.tasks) + 1)]
        workers = []

        for i, task_cfg in enumerate(self.config.tasks):
            # Instantiate task processor with dependency injection
            processor = self._instantiate_task(task_cfg)

            # Create worker
            input_q = queues[i]
            output_q = queues[i + 1] if i < len(self.config.tasks) else None

            worker = Worker(name=task_cfg.name, input_q=input_q, output_q=output_q, process_fn=processor)
            workers.append(worker)
            logger.debug(f"Created worker: {task_cfg.name}")

        return workers, queues

    def _instantiate_task(self, task_cfg) -> TaskProcessor:
        """
        Instantiate a task processor with dependency injection.

        Args:
            task_cfg: Task configuration

        Returns:
            TaskProcessor instance

        Raises:
            KeyError: If task type not found
            TypeError: If instantiation fails
        """
        # Try to get from registry first
        try:
            task_class = TaskRegistry.get(task_cfg.task_type)
            logger.debug(f"Found task '{task_cfg.task_type}' in registry")
        except KeyError:
            # If not in registry, try dynamic import
            logger.debug(f"Task '{task_cfg.task_type}' not in registry, trying dynamic import")
            task_class = self._import_task_class(task_cfg.task_type)

        # Resolve parameters with dependency injection
        resolved_params = self._resolve_params(task_cfg.params)

        # Instantiate task
        try:
            instance = task_class(**resolved_params)
            logger.debug(f"Instantiated task: {task_cfg.name}")
            return instance
        except TypeError as e:
            raise TypeError(f"Failed to instantiate task '{task_cfg.name}' " f"of type '{task_cfg.task_type}': {e}")

    def _import_task_class(self, class_path: str):
        """
        Dynamically import a class from a fully qualified path.

        Args:
            class_path: Fully qualified class name (e.g., "tasks.ocr.UploadTask")

        Returns:
            The class object

        Raises:
            ImportError: If module or class not found
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Could not import task class '{class_path}': {e}")

    def _resolve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve parameters with dependency injection.
        Parameters starting with "@" are looked up in the providers registry.

        Args:
            params: Raw parameters from configuration

        Returns:
            Resolved parameters

        Example:
            Input: {"uploader": "@nvidia_uploader", "timeout": 30}
            Output: {"uploader": <NvidiaAssetUploader instance>, "timeout": 30}
        """
        resolved = {}

        for key, value in params.items():
            if isinstance(value, str) and value.startswith("@"):
                # Reference to registered provider
                provider_name = value[1:]
                if provider_name not in self.providers:
                    raise KeyError(
                        f"Provider '{provider_name}' not registered. "
                        f"Available providers: {list(self.providers.keys())}"
                    )
                resolved[key] = self.providers[provider_name]
                logger.debug(f"Injected provider '{provider_name}' for parameter '{key}'")
            else:
                resolved[key] = value

        return resolved
