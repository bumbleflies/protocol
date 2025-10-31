import queue
import re
from pathlib import Path

from tasks import FileTask, FinalizeTask


class FileLoader:
    def __init__(self, target_q: queue.Queue, input_dir: str = ".", extension: str = ".jpg") -> None:
        self.input_dir: Path = Path(input_dir)
        self.extension: str = extension.lower()
        self.target_q: queue.Queue = target_q

    def _compute_sort_key(self, file_path: Path) -> float | None:
        """
        Extract leading number and optional sub-number for sorting.
        Examples:
          01.1_somefile.jpg -> 1.1
          01.2_somefile.jpg -> 1.2
          02_file.jpg      -> 2.0
        Returns None if no number is detected.
        """
        match: re.Match | None = re.match(r"^(\d+)(?:\.(\d+))?_?", file_path.stem)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) else 0
            return major + minor / 100.0  # minor as fractional
        return None

    def load_files(self) -> None:
        """Scan input dir, compute sort key, push FileTask into queue."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        files: list[Path] = [f for f in self.input_dir.glob(f"*{self.extension}") if f.is_file()]

        # Sort files: numbered files by their sort key, unnumbered files by name
        def sort_key_fn(f: Path) -> tuple:
            key = self._compute_sort_key(f)
            if key is None:
                # Unnumbered files: sort by name after all numbered files
                return (1, f.name)
            else:
                # Numbered files: sort by numeric key
                return (0, key)

        files.sort(key=sort_key_fn)

        # Find the highest numbered file to continue counter from
        max_key = 0.0
        for f in files:
            key = self._compute_sort_key(f)
            if key is not None:
                max_key = max(max_key, key)

        # Assign sequential sort keys, using extracted number or counter
        counter = int(max_key) + 1  # Start counter after last numbered file
        for f in files:
            extracted_key = self._compute_sort_key(f)
            if extracted_key is not None:
                sort_key = extracted_key
            else:
                sort_key = float(counter)
                counter += 1
            task: FileTask = FileTask(file_path=f, sort_key=sort_key)
            self.target_q.put(task)

        # Push finalize task
        self.target_q.put(FinalizeTask())
