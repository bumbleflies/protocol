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

    def run(self) -> None:
        logger.debug("Worker started")
        while not self._stop_event.is_set():
            try:
                task: Union[FileTask, FinalizeTask, StatusTask] = self.input_q.get(timeout=0.1)
                logger.debug(f"Got task: {task}")
            except Empty:
                continue

            self.current_task = task
            result: Union[FileTask, FinalizeTask, StatusTask]
            try:
                if isinstance(task, StatusTask):
                    # Pass StatusTask through processor (some may populate it)
                    # Then log it if this is the last worker
                    result = self.process_fn.process(task) if hasattr(self.process_fn, "process") else task
                    if not self.output_q:
                        # This is the last worker - log the status
                        for message in task.messages:
                            logger.info(message)
                        if task.output_file:
                            logger.info(f"Pipeline completed. Processed {task.files_processed} file(s).")
                elif isinstance(task, FinalizeTask):
                    # Call finalize if processor supports it
                    if isinstance(self.process_fn, FinalizableTaskProcessor):
                        self.process_fn.finalize()
                    result = task  # Pass through FinalizeTask
                else:
                    result = self.process_fn.process(task)

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
