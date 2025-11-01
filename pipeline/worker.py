import logging
import threading
from queue import Queue, Empty
from typing import Optional, Union

from tasks import FileTask, FinalizeTask, StatusTask
from tasks.task_item import TaskProcessor, FinalizableTaskProcessor

logger = logging.getLogger(__name__)


class Worker(threading.Thread):
    def __init__(
        self,
        name: str,
        input_q: Queue[Union[FileTask, FinalizeTask, StatusTask]],
        output_q: Optional[Queue[Union[FileTask, FinalizeTask, StatusTask]]],
        process_fn: TaskProcessor,
    ) -> None:
        super().__init__(daemon=True, name=name)
        self.name: str = name
        self.input_q: Queue[Union[FileTask, FinalizeTask, StatusTask]] = input_q
        self.output_q: Optional[Queue[Union[FileTask, FinalizeTask, StatusTask]]] = output_q
        self.process_fn: TaskProcessor = process_fn
        self.current_task: Optional[Union[FileTask, FinalizeTask, StatusTask]] = None
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

    def _process_status_task(self, task: StatusTask) -> StatusTask:
        """
        Process a status task.

        Args:
            task: The status task to process

        Returns:
            The processed status task
        """
        # Pass StatusTask through processor (some may populate it)
        result = self.process_fn.process(task) if hasattr(self.process_fn, "process") else task

        # Log status if this is the last worker
        if not self.output_q:
            for message in task.messages:
                logger.info(message)
            if task.output_file:
                logger.info(f"Pipeline completed. Processed {task.files_processed} file(s).")

        # Type checker: process() returns Union[FileTask, StatusTask] but we know it's StatusTask
        return result  # type: ignore[return-value]

    def _process_finalize_task(self, task: FinalizeTask) -> FinalizeTask:
        """
        Process a finalize task.

        Args:
            task: The finalize task to process

        Returns:
            The finalize task (passed through)
        """
        # Call finalize if processor supports it
        if isinstance(self.process_fn, FinalizableTaskProcessor):
            self.process_fn.finalize()

        return task

    def _process_file_task(self, task: FileTask) -> FileTask:
        """
        Process a file task.

        Args:
            task: The file task to process

        Returns:
            The processed file task
        """
        # Type checker: process() returns Union[FileTask, StatusTask] but we know it's FileTask
        return self.process_fn.process(task)  # type: ignore[return-value]

    def _handle_task(
        self, task: Union[FileTask, FinalizeTask, StatusTask]
    ) -> Union[FileTask, FinalizeTask, StatusTask]:
        """
        Handle a task by dispatching to the appropriate processor method.

        Args:
            task: The task to handle

        Returns:
            The processed task result
        """
        if isinstance(task, StatusTask):
            return self._process_status_task(task)
        elif isinstance(task, FinalizeTask):
            return self._process_finalize_task(task)
        else:
            return self._process_file_task(task)

    def _finalize_task_processing(
        self, task: Union[FileTask, FinalizeTask, StatusTask], result: Union[FileTask, FinalizeTask, StatusTask]
    ) -> None:
        """
        Finalize task processing by updating counters, signaling events, and forwarding results.

        Args:
            task: The original task
            result: The processed task result
        """
        self.current_task = None
        self.done_count += 1

        if isinstance(task, FinalizeTask):
            logger.debug("Finalize task processed, signaling completion")
            self.finalize_done_event.set()

        if self.output_q:
            self.output_q.put(result)
            logger.debug(f"Task forwarded to output queue: {result}")

        self.input_q.task_done()

    def run(self) -> None:
        logger.debug("Worker started")
        while not self._stop_event.is_set():
            try:
                task: Union[FileTask, FinalizeTask, StatusTask] = self.input_q.get(timeout=0.1)
                logger.debug(f"Got task: {task}")
            except Empty:
                continue

            self.current_task = task
            try:
                result = self._handle_task(task)
                logger.debug(f"Processed task: {task}")
            except Exception as e:
                logger.exception(f"Error processing task {task}: {e}")
                result = task  # fallback to pass along the original task

            self._finalize_task_processing(task, result)

        logger.debug("Worker exiting")
