import cv2
import numpy as np
from typing import Optional

from .task_item import FileTask, TaskProcessor
from .registry import TaskRegistry


@TaskRegistry.register("image_optimization")
class ImageOptimizationTask(TaskProcessor):
    """
    Detects and crops flipcharts using quadrilateral detection.
    Handles perspective correction for angled shots.

    This approach detects the flipchart boundary (edges of the paper/board)
    rather than the content boundaries, making it robust to:
    - Background clutter (walls, shelves, blackboards)
    - Camera UI overlays
    - Empty or partially filled flipcharts
    - Angled/perspective shots
    """

    def __init__(
        self,
        target_width: int = 1920,
        min_area_ratio: float = 0.10,
        max_area_ratio: float = 0.95,
        enable_perspective_correction: bool = True,
    ):
        """
        Args:
            target_width: Target width for output image (maintains aspect ratio)
            min_area_ratio: Minimum flipchart area as ratio of image (filters noise)
            max_area_ratio: Maximum area ratio (avoid detecting entire image)
            enable_perspective_correction: Apply perspective transform for angled shots
        """
        self.target_width = target_width
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.enable_perspective_correction = enable_perspective_correction

    def process(self, task: FileTask) -> FileTask:
        """
        Process a FileTask by detecting and cropping the flipchart.

        Args:
            task: The FileTask containing the image path

        Returns:
            The FileTask with optimized image in the img field

        Raises:
            ValueError: If image cannot be read
        """
        img = cv2.imread(str(task.file_path))
        if img is None:
            raise ValueError(f"Could not read image {task.file_path}")

        # Detect flipchart quadrilateral
        quad = self._detect_flipchart_quad(img)

        if quad is not None:
            if self.enable_perspective_correction:
                # Apply perspective correction (deskew angled shots)
                img = self._apply_perspective_transform(img, quad)
            else:
                # Simple crop using bounding rectangle
                x, y, w, h = cv2.boundingRect(quad)
                img = img[y : y + h, x : x + w]
        else:
            # Fallback: aggressive edge-based cropping
            img = self._fallback_edge_crop(img)

        # Resize to target width while maintaining aspect ratio
        if img.shape[1] > self.target_width:
            ratio = self.target_width / img.shape[1]
            new_h = int(img.shape[0] * ratio)
            img = cv2.resize(img, (self.target_width, new_h), interpolation=cv2.INTER_AREA)

        task.img = img
        return task

    def _detect_flipchart_quad(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the flipchart quadrilateral using edge detection.

        Uses Canny edge detection followed by contour analysis to find
        the rectangular boundary of the flipchart/paper.

        Args:
            img: Input BGR image

        Returns:
            4-point contour of the flipchart, or None if not found
        """
        h, w = img.shape[:2]
        image_area = h * w

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Edge detection with Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological operations to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the first quadrilateral that meets our criteria
        for contour in contours[:10]:  # Check top 10 largest contours
            area = cv2.contourArea(contour)
            area_ratio = area / image_area

            # Filter by area ratio
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Check if it's a quadrilateral (4 corners)
            if len(approx) == 4:
                # Verify it's roughly rectangular (aspect ratio check)
                if self._is_valid_quad(approx, w, h):
                    return approx

        return None

    def _is_valid_quad(self, quad: np.ndarray, img_w: int, img_h: int) -> bool:
        """
        Validate that the quadrilateral is a reasonable flipchart shape.

        Checks:
        - Aspect ratio is reasonable (0.5 - 2.5)
        - Quad is not just the entire image border

        Args:
            quad: 4-point contour
            img_w: Image width
            img_h: Image height

        Returns:
            True if quad is valid flipchart shape
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(quad)

        # Aspect ratio check (flipcharts are usually wider than tall or roughly square)
        aspect_ratio = w / h if h > 0 else 0
        if not (0.5 <= aspect_ratio <= 2.5):
            return False

        # Check that quad isn't too close to image edges (likely entire image)
        margin = 20
        if x < margin and y < margin and (x + w) > (img_w - margin) and (y + h) > (img_h - margin):
            return False

        return True

    def _apply_perspective_transform(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to correct angled shots.

        Takes a quadrilateral and transforms it to a rectangular front-facing view.
        This is the same technique used by document scanning apps.

        Args:
            img: Input image
            quad: 4-point contour of flipchart

        Returns:
            Perspective-corrected image (deskewed)
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = self._order_points(quad.reshape(4, 2))

        # Calculate dimensions of the output image
        (tl, tr, br, bl) = pts

        # Width: max of top and bottom edge lengths
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Height: max of left and right edge lengths
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the perspective transform (rectangular)
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32"
        )

        # Calculate perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in consistent order: top-left, top-right, bottom-right, bottom-left.

        Uses sum and diff of coordinates to determine corner positions.

        Args:
            pts: 4x2 array of points in any order

        Returns:
            Ordered 4x2 array: [top-left, top-right, bottom-right, bottom-left]
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left has smallest sum (x+y), bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        # Top-right has smallest diff (x-y), bottom-left has largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def _fallback_edge_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Fallback method: crop based on detected edges with generous margins.

        Used when quadrilateral detection fails (e.g., edges not clearly visible).

        Args:
            img: Input image

        Returns:
            Cropped image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 150)

        # Find non-zero edge pixels
        coords = cv2.findNonZero(edges)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Add generous margin (10% of dimensions)
            margin_w = int(w * 0.1)
            margin_h = int(h * 0.1)
            x = max(0, x - margin_w)
            y = max(0, y - margin_h)
            w = min(img.shape[1] - x, w + 2 * margin_w)
            h = min(img.shape[0] - y, h + 2 * margin_h)

            return img[y : y + h, x : x + w]

        # Last resort: return original image
        return img
