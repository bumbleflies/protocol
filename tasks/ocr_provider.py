from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

import numpy as np

from .task_item import OCRBox


class AssetUploader(ABC):
    """
    Abstract interface for uploading images to cloud storage.
    Implementations should handle the specifics of different cloud providers.
    """

    @abstractmethod
    def upload(self, image_bytes: bytes, description: str) -> UUID:
        """
        Upload image bytes to cloud storage.

        Args:
            image_bytes: The image data as bytes
            description: A description or filename for the asset

        Returns:
            A unique identifier (UUID) for the uploaded asset

        Raises:
            Exception: If upload fails
        """
        pass


class OCRProvider(ABC):
    """
    Abstract interface for OCR (Optical Character Recognition) services.
    Implementations should handle the specifics of different OCR providers.
    """

    @abstractmethod
    def detect_text(self, asset_id: UUID, image: np.ndarray) -> List[OCRBox]:
        """
        Perform OCR on an uploaded image asset.

        Args:
            asset_id: The unique identifier of the uploaded asset
            image: The image array (for providers that need it)

        Returns:
            List of OCRBox objects containing detected text and bounding boxes

        Raises:
            Exception: If OCR processing fails
        """
        pass
