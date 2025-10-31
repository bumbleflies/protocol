import logging
import threading
import time
from typing import List

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .worker import Worker

logger = logging.getLogger(__name__)


class WorkflowMonitor(threading.Thread):
    def __init__(self, workers: List[Worker], refresh_interval: float = 0.5) -> None:
        super().__init__(daemon=False, name="WorkflowMonitor")  # Changed to non-daemon for proper cleanup
        self.workers: List[Worker] = workers
        self.refresh_interval: float = refresh_interval
        self.console: Console = Console()
        self._stop_event: threading.Event = threading.Event()
        self._last_idle_logged = {w.name: False for w in workers}

    def stop(self) -> None:
        """Signal the monitor thread to stop."""
        self._stop_event.set()

    def render(self) -> Table:
        """Build and return the Rich table for the pipeline status."""
        table: Table = Table(title="Workflow Pipeline Monitor")
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Processing", style="magenta", no_wrap=True, width=90, max_width=100)
        table.add_column("Queued", style="yellow", justify="center")
        table.add_column("Done", style="green", justify="center")

        for w in self.workers:
            processing: str = str(w.current_task) if w.current_task else "-"
            done: str = str(w.done_count)
            queue_size: str = str(w.input_q.qsize())
            table.add_row(w.name, processing, queue_size, done)

        return table

    def run(self) -> None:
        """Continuously refresh the live monitor display and log worker status."""
        with Live(self.render(), refresh_per_second=4, console=self.console) as live:
            while not self._stop_event.is_set():
                time.sleep(self.refresh_interval)
                live.update(self.render())

                for w in self.workers:
                    current = str(w.current_task) if w.current_task else "-"
                    logger.debug(
                        f"Worker '{w.name}': Processing={current}, " f"Queued={w.input_q.qsize()}, Done={w.done_count}"
                    )
                    # Log idle once when it becomes idle
                    if w.current_task is None and not self._last_idle_logged[w.name]:
                        logger.debug(f"Worker '{w.name}' is now idle")
                        self._last_idle_logged[w.name] = True
                    elif w.current_task is not None:
                        self._last_idle_logged[w.name] = False

        # Final update before exit
        live.update(self.render())
        for w in self.workers:
            current = str(w.current_task) if w.current_task else "-"
            logger.debug(
                f"[FINAL] Worker '{w.name}': Processing={current}, " f"Queued={w.input_q.qsize()}, Done={w.done_count}"
            )
