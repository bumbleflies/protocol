import logging
import threading
from queue import Queue, Empty
from typing import Callable, Optional, Union

from tasks import FileTask, FinalizeTask

logger = logging.getLogger(__name__)


class Worker(threading.Thread):
    def __init__(
            self,
            name: str,
            input_q: Queue[Union[FileTask, FinalizeTask]],
            output_q: Optional[Queue[FileTask]],
            process_fn: Callable[[Union[FileTask, FinalizeTask]], Union[FileTask, FinalizeTask]]
    ) -> None:
        super().__init__(daemon=True, name=name)
        self.name: str = name
        self.input_q: Queue[Union[FileTask, FinalizeTask]] = input_q
        self.output_q: Optional[Queue[FileTask]] = output_q
        self.process_fn: Callable[[Union[FileTask, FinalizeTask]], Union[FileTask, FinalizeTask]] = process_fn
        self.current_task: Optional[Union[FileTask, FinalizeTask]] = None
        self.done_count: int = 0
        self._stop_event: threading.Event = threading.Event()
        self.finalize_done_event: threading.Event = threading.Event()  # signals finalize task done

    def stop(self) -> None:
        logger.debug("Stop signal received")
        self.finalize_done_event.wait()
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
                result: Union[FileTask, FinalizeTask] = self.process_fn(task)
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
