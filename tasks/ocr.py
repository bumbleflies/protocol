import logging
from typing import Union

import cv2

from .ocr_provider import AssetUploader, OCRProvider
from .task_item import TaskProcessor, StatusTask, FileTask
from .registry import TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("upload")
class UploadTask(TaskProcessor):
    """
    Uploads images to cloud storage using an AssetUploader provider.
    This task is now provider-agnostic and supports any AssetUploader implementation.
    """

    def __init__(self, uploader: AssetUploader):
        """
        Initialize the upload task with an asset uploader.

        Args:
            uploader: An AssetUploader implementation (e.g., NvidiaAssetUploader)
        """
        self.uploader = uploader

    def process(self, task: Union[FileTask, StatusTask]) -> Union[FileTask, StatusTask]:
        """
        Process a FileTask or StatusTask.

        Args:
            task: The FileTask containing the image to upload, or StatusTask to pass through

        Returns:
            The FileTask with asset_id populated, or StatusTask unchanged

        Raises:
            Exception: If upload fails
            TypeError: If task is not FileTask or StatusTask
        """
        # Pass through StatusTask unchanged
        if isinstance(task, StatusTask):
            return task

        if not isinstance(task, FileTask):
            raise TypeError(f"Expected FileTask or StatusTask, got {type(task).__name__}")

        logger.debug(f"Uploading image for task: {task.file_path}")
        try:
            _, img_bytes = cv2.imencode(".jpg", task.img)
            task.asset_id = self.uploader.upload(img_bytes.tobytes(), f"{task.file_path.name}")
            logger.debug(f"Upload completed for {task.file_path.name}, asset_id={task.asset_id}")
        except Exception as e:
            logger.exception(f"UploadTask failed for {task.file_path}: {e}")
            raise

        return task


@TaskRegistry.register("ocr")
class OCRTask(TaskProcessor):
    """
    Performs OCR using an OCRProvider.
    This task is now provider-agnostic and supports any OCRProvider implementation.
    """

    def __init__(self, provider: OCRProvider):
        """
        Initialize the OCR task with an OCR provider.

        Args:
            provider: An OCRProvider implementation (e.g., NvidiaOCRProvider)
        """
        self.provider = provider

    def process(self, task: Union[FileTask, StatusTask]) -> Union[FileTask, StatusTask]:
        """
        Process a FileTask or StatusTask.

        Args:
            task: The FileTask with asset_id set, or StatusTask to pass through

        Returns:
            The FileTask with ocr_boxes populated, or StatusTask unchanged

        Raises:
            ValueError: If asset_id is not set
            Exception: If OCR fails
            TypeError: If task is not FileTask or StatusTask
        """
        # Pass through StatusTask unchanged
        if isinstance(task, StatusTask):
            return task

        if not isinstance(task, FileTask):
            raise TypeError(f"Expected FileTask or StatusTask, got {type(task).__name__}")

        if not hasattr(task, "asset_id") or task.asset_id is None:
            msg = f"Task {task.file_path} has no asset_id; upload must run first"
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(f"Performing OCR for task: {task.file_path}")
        try:
            task.ocr_boxes = self.provider.detect_text(task.asset_id, task.img)
            logger.debug(f"OCR completed for {task.file_path}, found {len(task.ocr_boxes)} text boxes")
        except Exception as e:
            logger.exception(f"OCRTask failed for {task.file_path}: {e}")
            raise

        return task
