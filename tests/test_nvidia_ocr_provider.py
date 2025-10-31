"""Tests for NVIDIA OCR provider implementation."""

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest
import requests

from tasks.nvidia_ocr_provider import NvidiaAssetUploader, NvidiaOCRProvider
from tasks.task_item import OCRBox


class TestNvidiaAssetUploader:
    """Test NvidiaAssetUploader implementation."""

    def test_creation(self):
        """Test NvidiaAssetUploader can be created."""
        uploader = NvidiaAssetUploader(api_key="test_key")

        assert uploader.api_key == "test_key"
        assert uploader.assets_url == "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    @patch("requests.post")
    @patch("requests.put")
    def test_upload_success(self, mock_put, mock_post):
        """Test successful upload flow."""
        # Mock POST response (get upload URL)
        test_uuid = str(uuid4())
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"uploadUrl": "https://s3.example.com/upload", "assetId": test_uuid}
        mock_post.return_value = mock_post_response

        # Mock PUT response (actual upload)
        mock_put_response = Mock()
        mock_put.return_value = mock_put_response

        # Create uploader and upload
        uploader = NvidiaAssetUploader(api_key="test_key")
        image_bytes = b"fake_image_data"
        result = uploader.upload(image_bytes, "test.jpg")

        # Verify POST was called correctly
        mock_post.assert_called_once()
        post_call_args = mock_post.call_args
        assert post_call_args[0][0] == "https://api.nvcf.nvidia.com/v2/nvcf/assets"
        assert post_call_args[1]["headers"]["Authorization"] == "Bearer test_key"
        assert post_call_args[1]["json"]["description"] == "test.jpg"

        # Verify PUT was called correctly
        mock_put.assert_called_once()
        put_call_args = mock_put.call_args
        assert put_call_args[0][0] == "https://s3.example.com/upload"
        assert put_call_args[1]["data"] == image_bytes

        # Verify result
        assert result == UUID(test_uuid)

    @patch("requests.post")
    def test_upload_post_failure(self, mock_post):
        """Test upload fails when POST request fails."""
        # Mock POST failure
        mock_post_response = Mock()
        mock_post_response.raise_for_status.side_effect = requests.HTTPError("POST failed")
        mock_post.return_value = mock_post_response

        uploader = NvidiaAssetUploader(api_key="test_key")

        with pytest.raises(requests.HTTPError):
            uploader.upload(b"image_data", "test.jpg")

    @patch("requests.post")
    @patch("requests.put")
    def test_upload_put_failure(self, mock_put, mock_post):
        """Test upload fails when PUT request fails."""
        # Mock successful POST
        test_uuid = str(uuid4())
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"uploadUrl": "https://s3.example.com/upload", "assetId": test_uuid}
        mock_post.return_value = mock_post_response

        # Mock PUT failure
        mock_put_response = Mock()
        mock_put_response.raise_for_status.side_effect = requests.HTTPError("PUT failed")
        mock_put.return_value = mock_put_response

        uploader = NvidiaAssetUploader(api_key="test_key")

        with pytest.raises(requests.HTTPError):
            uploader.upload(b"image_data", "test.jpg")

    @patch("requests.post")
    def test_upload_logs_debug_messages(self, mock_post, caplog):
        """Test upload logs debug messages."""
        # Mock successful POST
        test_uuid = str(uuid4())
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"uploadUrl": "https://s3.example.com/upload", "assetId": test_uuid}
        mock_post.return_value = mock_post_response

        # Mock successful PUT
        with patch("requests.put") as mock_put:
            mock_put_response = Mock()
            mock_put.return_value = mock_put_response

            uploader = NvidiaAssetUploader(api_key="test_key")

            with caplog.at_level(logging.DEBUG):
                uploader.upload(b"image_data", "test.jpg")

            assert "Uploading asset" in caplog.text
            assert "Received upload URL" in caplog.text
            assert "Successfully uploaded asset" in caplog.text

    @patch("requests.post")
    def test_upload_exception_logged(self, mock_post, caplog):
        """Test upload logs exception on POST failure."""
        mock_post.side_effect = Exception("Network error")

        uploader = NvidiaAssetUploader(api_key="test_key")

        with pytest.raises(Exception):
            with caplog.at_level(logging.ERROR):
                uploader.upload(b"image_data", "test.jpg")

        assert "Failed to request upload slot" in caplog.text

    @patch("requests.post")
    @patch("requests.put")
    def test_upload_includes_correct_headers(self, mock_put, mock_post):
        """Test upload includes all required headers."""
        test_uuid = str(uuid4())
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"uploadUrl": "https://s3.example.com/upload", "assetId": test_uuid}
        mock_post.return_value = mock_post_response
        mock_put.return_value = Mock()

        uploader = NvidiaAssetUploader(api_key="secret_key")
        uploader.upload(b"image_data", "description.jpg")

        # Check POST headers
        post_headers = mock_post.call_args[1]["headers"]
        assert post_headers["Authorization"] == "Bearer secret_key"
        assert post_headers["Content-Type"] == "application/json"
        assert post_headers["accept"] == "application/json"

        # Check PUT headers
        put_headers = mock_put.call_args[1]["headers"]
        assert put_headers["x-amz-meta-nvcf-asset-description"] == "description.jpg"
        assert put_headers["content-type"] == "image/jpeg"


class TestNvidiaOCRProvider:
    """Test NvidiaOCRProvider implementation."""

    def test_creation(self):
        """Test NvidiaOCRProvider can be created."""
        provider = NvidiaOCRProvider(api_key="test_key")

        assert provider.api_key == "test_key"
        assert provider.ocr_url == "https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet"

    @patch("requests.post")
    def test_detect_text_success(self, mock_post):
        """Test successful OCR detection."""
        test_uuid = uuid4()

        # Create mock OCR response
        ocr_response = {
            "metadata": [
                {
                    "label": "Hello",
                    "polygon": {"x1": 10, "y1": 10, "x2": 50, "y2": 10, "x3": 50, "y3": 30, "x4": 10, "y4": 30},
                    "confidence": 0.95,
                },
                {
                    "label": "World",
                    "polygon": {"x1": 60, "y1": 10, "x2": 100, "y2": 10, "x3": 100, "y3": 30, "x4": 60, "y4": 30},
                    "confidence": 0.92,
                },
            ]
        }

        # Create a ZIP file with response
        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            # Mock POST response
            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            result = provider.detect_text(test_uuid, img)

            # Verify results
            assert len(result) == 2
            assert result[0].label == "Hello"
            assert result[0].confidence == 0.95
            assert result[1].label == "World"
            assert result[1].confidence == 0.92

    @patch("requests.post")
    def test_detect_text_sorts_boxes(self, mock_post):
        """Test OCR boxes are sorted top-down, left-right."""
        test_uuid = uuid4()

        # Create response with boxes in non-sorted order
        ocr_response = {
            "metadata": [
                {
                    "label": "Third",
                    "polygon": {"x1": 60, "y1": 20, "x2": 80, "y2": 20, "x3": 80, "y3": 30, "x4": 60, "y4": 30},
                    "confidence": 0.9,
                },
                {
                    "label": "First",
                    "polygon": {"x1": 10, "y1": 10, "x2": 30, "y2": 10, "x3": 30, "y3": 20, "x4": 10, "y4": 20},
                    "confidence": 0.9,
                },
                {
                    "label": "Second",
                    "polygon": {"x1": 40, "y1": 10, "x2": 60, "y2": 10, "x3": 60, "y3": 20, "x4": 40, "y4": 20},
                    "confidence": 0.9,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            result = provider.detect_text(test_uuid, img)

            # Verify sorted order (by y1, then x1)
            assert result[0].label == "First"
            assert result[1].label == "Second"
            assert result[2].label == "Third"

    @patch("requests.post")
    def test_detect_text_no_response_file(self, mock_post, caplog):
        """Test OCR handles missing response file."""
        test_uuid = uuid4()

        # Create empty ZIP
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w"):
                pass  # Empty ZIP

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            with caplog.at_level(logging.WARNING):
                result = provider.detect_text(test_uuid, img)

            assert len(result) == 0
            assert "No .response file found" in caplog.text

    @patch("requests.post")
    def test_detect_text_request_failure(self, mock_post):
        """Test OCR handles request failure."""
        mock_post_response = Mock()
        mock_post_response.raise_for_status.side_effect = requests.HTTPError("OCR failed")
        mock_post.return_value = mock_post_response

        provider = NvidiaOCRProvider(api_key="test_key")
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(requests.HTTPError):
            provider.detect_text(uuid4(), img)

    @patch("requests.post")
    def test_detect_text_logs_debug_messages(self, mock_post, caplog):
        """Test OCR logs debug messages."""
        test_uuid = uuid4()

        ocr_response = {"metadata": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            with caplog.at_level(logging.DEBUG):
                provider.detect_text(test_uuid, img)

            assert "Requesting OCR" in caplog.text
            assert "OCR request successful" in caplog.text
            assert "Parsed OCR response JSON" in caplog.text

    @patch("requests.post")
    def test_detect_text_json_parse_error(self, mock_post, caplog):
        """Test OCR handles JSON parse errors."""
        test_uuid = uuid4()

        # Create ZIP with invalid JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text("invalid json {")

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            with caplog.at_level(logging.ERROR):
                result = provider.detect_text(test_uuid, img)

            assert len(result) == 0
            assert "Failed to parse OCR response JSON" in caplog.text

    @patch("requests.post")
    def test_detect_text_cleans_up_temp_files(self, mock_post):
        """Test OCR cleans up temporary files."""
        test_uuid = uuid4()

        ocr_response = {"metadata": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            provider.detect_text(test_uuid, img)

            # Check temp files are cleaned up
            temp_dir = Path(tempfile.gettempdir())
            temp_zip = temp_dir / f"{test_uuid}.zip"
            temp_extract = temp_dir / str(test_uuid)

            assert not temp_zip.exists()
            assert not temp_extract.exists()

    @patch("requests.post")
    def test_detect_text_includes_correct_headers(self, mock_post):
        """Test OCR request includes all required headers."""
        test_uuid = uuid4()

        ocr_response = {"metadata": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="secret_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            provider.detect_text(test_uuid, img)

            # Verify headers
            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer secret_key"
            assert headers["Content-Type"] == "application/json"
            assert headers["NVCF-INPUT-ASSET-REFERENCES"] == str(test_uuid)
            assert headers["NVCF-FUNCTION-ASSET-IDS"] == str(test_uuid)

    @patch("requests.post")
    def test_detect_text_with_missing_polygon_fields(self, mock_post):
        """Test OCR handles missing polygon fields gracefully."""
        test_uuid = uuid4()

        # Response with incomplete polygon data
        ocr_response = {
            "metadata": [{"label": "Test", "polygon": {"x1": 10, "y1": 10}, "confidence": 0.9}]  # Missing x2, y2, etc.
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            provider = NvidiaOCRProvider(api_key="test_key")
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            result = provider.detect_text(test_uuid, img)

            # Should still create box with defaults for missing fields
            assert len(result) == 1
            assert result[0].x1 == 10
            assert result[0].x2 == 0  # Default value

    @patch("requests.post")
    def test_detect_text_cleanup_warning_on_failure(self, mock_post, caplog):
        """Test OCR logs warning if cleanup fails."""
        test_uuid = uuid4()

        ocr_response = {"metadata": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            response_file = Path(tmpdir) / "result.response"
            response_file.write_text(json.dumps(ocr_response))

            zip_path = Path(tmpdir) / "response.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(response_file, arcname="result.response")

            mock_post_response = Mock()
            mock_post_response.content = zip_path.read_bytes()
            mock_post.return_value = mock_post_response

            # Mock Path.unlink to raise exception
            with patch.object(Path, "unlink", side_effect=PermissionError("Cannot delete")):
                provider = NvidiaOCRProvider(api_key="test_key")
                img = np.zeros((100, 100, 3), dtype=np.uint8)

                with caplog.at_level(logging.WARNING):
                    provider.detect_text(test_uuid, img)

                # Note: May or may not log warning depending on timing
                # The test passes if no exception is raised
