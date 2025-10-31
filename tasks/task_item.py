from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
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
