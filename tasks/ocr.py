import json
import logging
import os
import uuid
import zipfile
from pathlib import Path
from typing import Union

import cv2
import requests
from dotenv import load_dotenv

from .task_item import FileTask, FinalizeTask, OCRBox

logger = logging.getLogger(__name__)
load_dotenv()

NVIDIA_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet"
HEADER_AUTH = f"Bearer {os.getenv('NVIDIA_API_KEY')}"


def _upload_asset(image_bytes: bytes, description: str) -> uuid.UUID:
    """Upload image to NVCF and return asset UUID."""
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    logger.debug(f"Uploading asset '{description}' to {assets_url}")

    headers = {
        "Authorization": HEADER_AUTH,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }

    payload = {"contentType": "image/jpeg", "description": description}

    try:
        response = requests.post(assets_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        upload_url = response.json()["uploadUrl"]
        asset_id = response.json()["assetId"]
        logger.debug(f"Received upload URL and asset_id={asset_id} for {description}")
    except Exception as e:
        logger.exception(f"Failed to request upload slot for {description}: {e}")
        raise

    try:
        response = requests.put(upload_url, data=image_bytes, headers=s3_headers, timeout=300)
        response.raise_for_status()
        logger.debug(f"Successfully uploaded asset {asset_id}")
    except Exception as e:
        logger.exception(f"Failed to upload asset {asset_id}: {e}")
        raise

    return uuid.UUID(asset_id)


class UploadTask:
    """Uploads images to NVIDIA OCR service."""

    def __call__(self, task: Union[FileTask, FinalizeTask]) -> Union[FileTask, FinalizeTask]:
        if isinstance(task, FinalizeTask):
            logger.debug("UploadTask received FinalizeTask, passing through.")
            return task

        logger.debug(f"Uploading image for task: {task.file_path}")
        try:
            _, img_bytes = cv2.imencode(".jpg", task.img)
            task.asset_id = _upload_asset(img_bytes.tobytes(), f"{task.file_path.name}")
            logger.debug(f"Upload completed for {task.file_path.name}, asset_id={task.asset_id}")
        except Exception as e:
            logger.exception(f"UploadTask failed for {task.file_path}: {e}")
            raise

        return task


class OCRTask:
    """Fetch OCR result from NVIDIA OCR service and extract structured text boxes."""

    def __call__(self, task: Union[FileTask, FinalizeTask]) -> Union[FileTask, FinalizeTask]:
        if isinstance(task, FinalizeTask):
            logger.debug("OCRTask received FinalizeTask, passing through.")
            return task

        if not hasattr(task, "asset_id"):
            msg = f"Task {task.file_path} has no asset_id; upload must run first"
            logger.error(msg)
            raise ValueError(msg)

        asset_list = str(task.asset_id)
        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": HEADER_AUTH,
        }
        inputs = {"image": asset_list, "render_label": False}

        logger.debug(f"Requesting OCR for asset_id={asset_list}")

        try:
            response = requests.post(NVIDIA_URL, headers=headers, json=inputs, timeout=300)
            response.raise_for_status()
            logger.debug(f"OCR request successful for asset_id={asset_list}")
        except Exception as e:
            logger.exception(f"OCR request failed for asset_id={asset_list}: {e}")
            raise

        # Save response ZIP
        zip_path = Path(f"/tmp/{task.asset_id}.zip")
        extract_dir = zip_path.with_suffix("")
        zip_path.write_bytes(response.content)
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        response_files = list(extract_dir.glob("*.response"))
        task.ocr_boxes.clear()

        if not response_files:
            logger.warning(f"No .response file found in OCR output for {task.file_path}")
            return task

        response_path = response_files[0]
        try:
            with open(response_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Parsed OCR response JSON for {task.file_path}")

            metadata = data.get("metadata", [])
            for label in metadata:
                poly = label.get("polygon", {})
                task.ocr_boxes.append(
                    OCRBox(
                        label=label.get("label", ""),
                        x1=poly.get("x1", 0),
                        y1=poly.get("y1", 0),
                        x2=poly.get("x2", 0),
                        y2=poly.get("y2", 0),
                        x3=poly.get("x3", 0),
                        y3=poly.get("y3", 0),
                        x4=poly.get("x4", 0),
                        y4=poly.get("y4", 0),
                        confidence=label.get("confidence", 0.0),
                    )
                )

            # Sort by top-down, left-right order
            task.ocr_boxes.sort(key=lambda b: (b.y1, b.x1))
            ocr_text = " ".join([b.label for b in task.ocr_boxes])
            logger.debug(
                f"OCR extracted {len(task.ocr_boxes)} boxes for {task.file_path}: "
                f"'{ocr_text[:80]}...'"
            )

        except Exception as e:
            logger.exception(f"Failed to parse OCR response JSON for {task.file_path}: {e}")

        return task
