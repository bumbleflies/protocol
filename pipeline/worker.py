import logging
import threading
from queue import Queue, Empty
from typing import Callable, Optional, Union

from tasks import FileTask, FinalizeTask
from tasks.task_item import TaskProcessor, FinalizableTaskProcessor

logger = logging.getLogger(__name__)


class Worker(threading.Thread):
    def __init__(
        self,
        name: str,
        input_q: Queue[Union[FileTask, FinalizeTask]],
        output_q: Optional[Queue[FileTask]],
        process_fn: Union[TaskProcessor, Callable[[Union[FileTask, FinalizeTask]], Union[FileTask, FinalizeTask]]],
    ) -> None:
        super().__init__(daemon=True, name=name)
        self.name: str = name
        self.input_q: Queue[Union[FileTask, FinalizeTask]] = input_q
        self.output_q: Optional[Queue[FileTask]] = output_q
        self.process_fn: Union[TaskProcessor, Callable] = process_fn
        self.current_task: Optional[Union[FileTask, FinalizeTask]] = None
        self.done_count: int = 0
        self._stop_event: threading.Event = threading.Event()
        self.finalize_done_event: threading.Event = threading.Event()  # signals finalize task done

    def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker thread gracefully.

        Args:
            timeout: Maximum time to wait for finalization (seconds)
        """
        logger.debug("Stop signal received")
        if not self.finalize_done_event.wait(timeout=timeout):
            logger.warning(f"Worker '{self.name}' finalize timeout after {timeout}s")
        self._stop_event.set()

    def run(self) -> None:
        logger.debug("Worker started")
        while not self._stop_event.is_set():
            try:
                task: Union[FileTask, FinalizeTask] = self.input_q.get(timeout=0.1)
                logger.debug(f"Got task: {task}")
            except Empty:
                continue

            self.current_task = task
            try:
                # Handle new TaskProcessor interface
                if isinstance(self.process_fn, TaskProcessor):
                    if isinstance(task, FinalizeTask):
                        # Call finalize if processor supports it
                        if isinstance(self.process_fn, FinalizableTaskProcessor):
                            self.process_fn.finalize()
                        result = task  # Pass through FinalizeTask
                    else:
                        result = self.process_fn.process(task)
                else:
                    # Backward compatibility: call as Callable
                    result = self.process_fn(task)

                logger.debug(f"Processed task: {task}")
            except Exception as e:
                logger.exception(f"Error processing task {task}: {e}")
                result = task  # fallback to pass along the original task

            self.current_task = None
            self.done_count += 1

            if isinstance(task, FinalizeTask):
                logger.debug("Finalize task processed, signaling completion")
                self.finalize_done_event.set()

            if self.output_q:
                self.output_q.put(result)
                logger.debug(f"Task forwarded to output queue: {result}")

            self.input_q.task_done()
        logger.debug("Worker exiting")
