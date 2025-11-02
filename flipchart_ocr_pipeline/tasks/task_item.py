from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union
from uuid import UUID

import numpy as np


@dataclass
class OCRBox:
    """Represents one OCR-detected text region."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    x4: int
    y4: int
    confidence: float


@dataclass
class FileTask:
    file_path: Path
    sort_key: float
    ocr_boxes: List[OCRBox] = field(default_factory=list)
    asset_id: Optional[UUID] = None
    img: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class FinalizeTask:
    """Special task to signal the end of the pipeline for PDF generation."""

    pass


@dataclass
class StatusTask:
    """
    Task that carries status information through the pipeline.
    Used to report what was processed and any important messages.
    """

    files_processed: int = 0
    output_file: Optional[Path] = None
    messages: List[str] = field(default_factory=list)


class TaskProcessor(ABC):
    """
    Abstract base class for all task processors.
    Processors handle FileTask and StatusTask objects.
    StatusTask should be passed through unchanged by most processors.
    """

    @abstractmethod
    def process(self, task: Union[FileTask, StatusTask]) -> Union[FileTask, StatusTask]:
        """
        Process a FileTask or StatusTask and return the result.

        Args:
            task: The FileTask or StatusTask to process

        Returns:
            The processed task (FileTask or StatusTask)
        """
        pass


class FinalizableTaskProcessor(TaskProcessor):
    """
    Abstract base class for processors that need finalization logic.
    Used for tasks that accumulate state and need to perform an action
    when the FinalizeTask signal is received.
    """

    @abstractmethod
    def finalize(self) -> None:
        """
        Called when FinalizeTask is received.
        Use this to perform cleanup, save accumulated data, etc.
        """
        pass
