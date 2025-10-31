import json
import logging
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import requests

from .ocr_provider import AssetUploader, OCRProvider
from .task_item import OCRBox

logger = logging.getLogger(__name__)


class NvidiaAssetUploader(AssetUploader):
    """
    NVIDIA NVCF asset uploader implementation.
    Handles uploading images to NVIDIA's cloud storage.
    """

    def __init__(self, api_key: str):
        """
        Initialize the NVIDIA asset uploader.

        Args:
            api_key: NVIDIA API key for authentication
        """
        self.api_key = api_key
        self.assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    def upload(self, image_bytes: bytes, description: str) -> uuid.UUID:
        """
        Upload image to NVIDIA NVCF and return asset UUID.

        Args:
            image_bytes: The image data as bytes
            description: A description or filename for the asset

        Returns:
            UUID of the uploaded asset

        Raises:
            requests.HTTPError: If upload fails
        """
        logger.debug(f"Uploading asset '{description}' to {self.assets_url}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        s3_headers = {
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg",
        }

        payload = {"contentType": "image/jpeg", "description": description}

        try:
            response = requests.post(self.assets_url, headers=headers, json=payload, timeout=30)
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


class NvidiaOCRProvider(OCRProvider):
    """
    NVIDIA OCDRNet OCR provider implementation.
    Handles text detection using NVIDIA's OCR service.
    """

    def __init__(self, api_key: str):
        """
        Initialize the NVIDIA OCR provider.

        Args:
            api_key: NVIDIA API key for authentication
        """
        self.api_key = api_key
        self.ocr_url = "https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet"

    def detect_text(self, asset_id: uuid.UUID, image: np.ndarray) -> List[OCRBox]:
        """
        Perform OCR on an uploaded NVIDIA asset.

        Args:
            asset_id: The UUID of the uploaded asset
            image: The image array (not used by NVIDIA API, but part of interface)

        Returns:
            List of OCRBox objects with detected text and bounding boxes

        Raises:
            requests.HTTPError: If OCR request fails
        """
        asset_list = str(asset_id)
        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": f"Bearer {self.api_key}",
        }
        inputs = {"image": asset_list, "render_label": False}

        logger.debug(f"Requesting OCR for asset_id={asset_list}")

        try:
            response = requests.post(self.ocr_url, headers=headers, json=inputs, timeout=300)
            response.raise_for_status()
            logger.debug(f"OCR request successful for asset_id={asset_list}")
        except Exception as e:
            logger.exception(f"OCR request failed for asset_id={asset_list}: {e}")
            raise

        # Save response ZIP to temp directory (platform-independent)
        temp_dir = Path(tempfile.gettempdir())
        zip_path = temp_dir / f"{asset_id}.zip"
        extract_dir = temp_dir / str(asset_id)

        zip_path.write_bytes(response.content)
        extract_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            response_files = list(extract_dir.glob("*.response"))
            ocr_boxes = []

            if not response_files:
                logger.warning(f"No .response file found in OCR output for asset {asset_id}")
                return ocr_boxes

            response_path = response_files[0]
            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"Parsed OCR response JSON for asset {asset_id}")

                metadata = data.get("metadata", [])
                for label in metadata:
                    poly = label.get("polygon", {})
                    ocr_boxes.append(
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
                ocr_boxes.sort(key=lambda b: (b.y1, b.x1))
                ocr_text = " ".join([b.label for b in ocr_boxes])
                logger.debug(
                    f"OCR extracted {len(ocr_boxes)} boxes for asset {asset_id}: "
                    f"'{ocr_text[:80]}...'"
                )

            except Exception as e:
                logger.exception(f"Failed to parse OCR response JSON for asset {asset_id}: {e}")

            return ocr_boxes

        finally:
            # Cleanup temp files
            try:
                if zip_path.exists():
                    zip_path.unlink()
                if extract_dir.exists():
                    for file in extract_dir.glob("*"):
                        file.unlink()
                    extract_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files for asset {asset_id}: {e}")
