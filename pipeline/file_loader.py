import queue
import re
from pathlib import Path

from tasks import FileTask, FinalizeTask


class FileLoader:
    def __init__(self, target_q: queue.Queue, input_dir: str = ".", extension: str = ".jpg") -> None:
        self.input_dir: Path = Path(input_dir)
        self.extension: str = extension.lower()
        self.target_q: queue.Queue = target_q

    def _compute_sort_key(self, file_path: Path) -> float:
        """
        Extract leading number and optional sub-number for sorting.
        Examples:
          01.1_somefile.jpg -> 1.1
          01.2_somefile.jpg -> 1.2
          02_file.jpg      -> 2.0
        """
        match: re.Match | None = re.match(r"^(\d+)(?:\.(\d+))?_?", file_path.stem)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return major + minor / 100.0  # minor as fractional
        return 0.0

    def load_files(self) -> None:
        """Scan input dir, compute sort key, push FileTask into queue."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        files: list[Path] = [f for f in self.input_dir.glob(f"*{self.extension}") if f.is_file()]
        files.sort(key=self._compute_sort_key)

        for f in files:
            task: FileTask = FileTask(file_path=f, sort_key=self._compute_sort_key(f))
            self.target_q.put(task)

        # Push finalize task
        self.target_q.put(FinalizeTask())
