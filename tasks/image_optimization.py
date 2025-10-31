import cv2
import numpy as np
from typing import Optional, Union

from .task_item import TaskProcessor, StatusTask, FileTask
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

    def process(self, task: Union[FileTask, StatusTask]) -> Union[FileTask, StatusTask]:
        """
        Process a FileTask or StatusTask.

        Args:
            task: The FileTask containing the image path, or StatusTask to pass through

        Returns:
            The FileTask with optimized image in the img field, or StatusTask unchanged

        Raises:
            ValueError: If image cannot be read
            TypeError: If task is not FileTask or StatusTask
        """
        # Pass through StatusTask unchanged
        if isinstance(task, StatusTask):
            return task

        if not isinstance(task, FileTask):
            raise TypeError(f"Expected FileTask or StatusTask, got {type(task).__name__}")

        img = cv2.imread(str(task.file_path))
        if img is None:
            raise ValueError(f"Could not read image {task.file_path}")

        # First, try to detect and correct rotation
        rotation_angle = self._detect_rotation_angle(img)
        if rotation_angle is not None and abs(rotation_angle) > 0.5:  # Only rotate if > 0.5 degrees
            img = self._rotate_image(img, rotation_angle)

        # Detect flipchart quadrilateral (works for flipcharts with clear edges)
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
            # Try brightness-based paper detection (works for white/bright paper)
            quad_bright = self._detect_bright_paper(img)
            if quad_bright is not None:
                if self.enable_perspective_correction:
                    img = self._apply_perspective_transform(img, quad_bright)
                else:
                    x, y, w, h = cv2.boundingRect(quad_bright)
                    img = img[y : y + h, x : x + w]
            else:
                # Last resort: aggressive edge-based cropping
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

        # Try multiple preprocessing strategies
        strategies = [
            # Strategy 1: Bilateral filter (preserves edges)
            lambda g: cv2.bilateralFilter(g, 9, 75, 75),
            # Strategy 2: Gaussian blur (good for noisy images)
            lambda g: cv2.GaussianBlur(g, (5, 5), 0),
            # Strategy 3: Median blur (removes salt-and-pepper noise)
            lambda g: cv2.medianBlur(g, 5),
        ]

        # Try multiple Canny thresholds
        canny_params = [
            (50, 150),  # Standard
            (30, 100),  # Lower thresholds for faint edges
            (75, 200),  # Higher thresholds for strong edges
        ]

        for blur_fn in strategies:
            blurred = blur_fn(gray)

            for low_thresh, high_thresh in canny_params:
                # Edge detection with Canny
                edges = cv2.Canny(blurred, low_thresh, high_thresh)

                # Dilate to connect nearby edges (helps with rotated/partial edges)
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                edges = cv2.dilate(edges, kernel_dilate, iterations=1)

                # Morphological closing to fill gaps
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=3)

                # Find contours
                contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                # Find the first quadrilateral that meets our criteria
                for contour in contours[:15]:  # Check top 15 largest contours
                    area = cv2.contourArea(contour)
                    area_ratio = area / image_area

                    # Filter by area ratio
                    if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                        continue

                    # Try multiple epsilon values for polygon approximation
                    perimeter = cv2.arcLength(contour, True)
                    for epsilon_factor in [0.015, 0.02, 0.025, 0.03, 0.04]:
                        approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)

                        # Check if it's a quadrilateral (4 corners)
                        if len(approx) == 4:
                            # Verify it's roughly rectangular (aspect ratio check)
                            if self._is_valid_quad(approx, w, h):
                                return approx

        return None

    def _detect_bright_paper(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect white/bright paper using brightness thresholding.

        This method works well for clean white flipchart paper on darker backgrounds
        where edge detection fails due to low contrast at paper edges.

        Args:
            img: Input BGR image

        Returns:
            4-point contour of the paper, or None if not found
        """
        h, w = img.shape[:2]
        image_area = h * w

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try multiple threshold values - lower values include more of background,
        # higher values isolate the brightest paper regions
        for threshold_value in [210, 200, 190, 180]:
            _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

            # Morphological closing to fill content holes and connect paper regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            # Find external contours (outermost boundaries only)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Find the first suitable quadrilateral
            for contour in contours[:5]:  # Check top 5 largest bright regions
                area = cv2.contourArea(contour)
                area_ratio = area / image_area

                # Paper should be significant portion of image (15-90%)
                # Lower threshold (15%) to catch tightly-cropped flipcharts
                if area_ratio < 0.15 or area_ratio > 0.90:
                    continue

                # Get convex hull first (straightens curved edges)
                hull = cv2.convexHull(contour)

                # Approximate hull to polygon
                perimeter = cv2.arcLength(hull, True)

                # Try different epsilon values to find a quadrilateral
                for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
                    approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)

                    if len(approx) == 4:
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
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

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

    def _detect_rotation_angle(self, img: np.ndarray) -> Optional[float]:
        """
        Detect rotation angle of the document using Hough line detection.

        Analyzes dominant lines in the image to determine if the document
        is rotated. Returns angle needed to correct rotation.

        Args:
            img: Input image

        Returns:
            Rotation angle in degrees (positive = counterclockwise), or None if can't detect
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(img.shape[:2]) // 4,
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return None

        # Calculate angles of all detected lines
        angles = []
        for line_data in lines:
            # HoughLinesP returns lines in format [[[x1, y1, x2, y2]]]
            line_coords = line_data[0]  # type: ignore[index]  # Extract [x1, y1, x2, y2]
            x1 = int(line_coords[0])  # type: ignore[index]
            y1 = int(line_coords[1])  # type: ignore[index]
            x2 = int(line_coords[2])  # type: ignore[index]
            y2 = int(line_coords[3])  # type: ignore[index]
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
            # Normalize to [-45, 45] range (considering 90-degree symmetry)
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

        if not angles:
            return None

        # Use median angle to avoid outliers
        median_angle = float(np.median(angles))

        # Only return angle if it's significant (> 0.5 degrees)
        if abs(median_angle) > 0.5:
            return -median_angle  # Negate to get correction angle
        return None

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle, expanding canvas to fit.

        Args:
            img: Input image
            angle: Rotation angle in degrees (positive = counterclockwise)

        Returns:
            Rotated image with expanded canvas
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust rotation matrix to account for translation
        rotation_matrix[0, 2] += new_w / 2 - center[0]
        rotation_matrix[1, 2] += new_h / 2 - center[1]

        # Perform rotation with white background
        rotated = cv2.warpAffine(
            img, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        return rotated
