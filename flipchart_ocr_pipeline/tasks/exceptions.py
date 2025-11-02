"""
Custom exceptions for the flipchart OCR pipeline.

This module defines a hierarchy of exceptions to provide consistent
error handling across the pipeline components.
"""


class PipelineException(Exception):
    """
    Base exception for all pipeline-related errors.

    All custom exceptions in the pipeline should inherit from this class
    to allow for easy catching of all pipeline-specific errors.
    """

    pass


class ImageProcessingException(PipelineException):
    """
    Exception raised when image processing operations fail.

    This includes failures in:
    - Image loading/reading
    - Image transformations (rotation, perspective correction, cropping)
    - Edge detection
    - Contour analysis
    """

    pass


class UploadException(PipelineException):
    """
    Exception raised when asset upload operations fail.

    This includes failures in:
    - Network connectivity to upload service
    - Authentication/authorization
    - Upload slot reservation
    - File upload
    """

    pass


class OCRException(PipelineException):
    """
    Exception raised when OCR operations fail.

    This includes failures in:
    - OCR service connectivity
    - OCR processing
    - Result parsing
    - Invalid OCR responses
    """

    pass


class ConfigurationException(PipelineException):
    """
    Exception raised when configuration is invalid or missing.

    This includes failures in:
    - Missing required configuration values
    - Invalid configuration format
    - Incompatible configuration options
    """

    pass


class FileOperationException(PipelineException):
    """
    Exception raised when file operations fail.

    This includes failures in:
    - File not found
    - Permission denied
    - Invalid file format
    - Disk I/O errors
    """

    pass
