import cv2
import numpy as np
from typing import Optional, Union

from .task_item import TaskProcessor, StatusTask, FileTask
from .registry import TaskRegistry
from .exceptions import ImageProcessingException


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

    # Rotation detection constants
    MIN_ROTATION_ANGLE_DEGREES = 0.5  # Minimum angle to trigger rotation correction

    # Edge detection preprocessing parameters
    BILATERAL_FILTER_DIAMETER = 9
    BILATERAL_FILTER_SIGMA_COLOR = 75
    BILATERAL_FILTER_SIGMA_SPACE = 75
    GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
    MEDIAN_BLUR_KERNEL_SIZE = 5

    # Canny edge detection thresholds (low, high)
    CANNY_THRESHOLD_STANDARD = (50, 150)
    CANNY_THRESHOLD_LOW = (30, 100)  # For faint edges
    CANNY_THRESHOLD_HIGH = (75, 200)  # For strong edges

    # Morphology kernel sizes
    DILATE_KERNEL_SIZE = (3, 3)
    CLOSE_KERNEL_SIZE = (7, 7)
    DILATE_ITERATIONS = 1
    CLOSE_ITERATIONS = 3

    # Contour analysis parameters
    MAX_CONTOURS_TO_ANALYZE = 15
    EPSILON_APPROXIMATION_FACTORS = [0.015, 0.02, 0.025, 0.03, 0.04]

    # Bright paper detection parameters
    BRIGHTNESS_THRESHOLDS = [210, 200, 190, 180]
    BRIGHT_PAPER_KERNEL_SIZE = (25, 25)
    MAX_BRIGHT_CONTOURS_TO_ANALYZE = 5
    BRIGHT_MIN_AREA_RATIO = 0.15
    BRIGHT_MAX_AREA_RATIO = 0.90
    BRIGHT_EPSILON_FACTORS = [0.02, 0.03, 0.04, 0.05, 0.06]

    # Quad validation parameters
    MIN_ASPECT_RATIO = 0.5
    MAX_ASPECT_RATIO = 2.5
    QUAD_VALIDATION_MARGIN = 20  # pixels

    # Edge-based cropping parameters
    EDGE_CROP_MARGIN_RATIO = 0.1  # 10% margin on each side
    EDGE_MIN_CROP_RATIO = 0.8  # Don't crop more than 20% of image

    # Hough line detection parameters
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 100
    HOUGH_MIN_LINE_LENGTH_RATIO = 0.25  # Fraction of image dimension
    HOUGH_MAX_LINE_GAP = 20

    # Canny parameters for rotation detection
    ROTATION_CANNY_LOW = 50
    ROTATION_CANNY_HIGH = 150
    ROTATION_CANNY_APERTURE = 3

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
            raise ImageProcessingException(f"Could not read image {task.file_path}")

        # First, try to detect and correct rotation
        rotation_angle = self._detect_rotation_angle(img)
        if rotation_angle is not None and abs(rotation_angle) > self.MIN_ROTATION_ANGLE_DEGREES:
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
            lambda g: cv2.bilateralFilter(
                g,
                self.BILATERAL_FILTER_DIAMETER,
                self.BILATERAL_FILTER_SIGMA_COLOR,
                self.BILATERAL_FILTER_SIGMA_SPACE,
            ),
            # Strategy 2: Gaussian blur (good for noisy images)
            lambda g: cv2.GaussianBlur(g, self.GAUSSIAN_BLUR_KERNEL_SIZE, 0),
            # Strategy 3: Median blur (removes salt-and-pepper noise)
            lambda g: cv2.medianBlur(g, self.MEDIAN_BLUR_KERNEL_SIZE),
        ]

        # Try multiple Canny thresholds
        canny_params = [
            self.CANNY_THRESHOLD_STANDARD,
            self.CANNY_THRESHOLD_LOW,
            self.CANNY_THRESHOLD_HIGH,
        ]

        for blur_fn in strategies:
            blurred = blur_fn(gray)

            for low_thresh, high_thresh in canny_params:
                # Edge detection with Canny
                edges = cv2.Canny(blurred, low_thresh, high_thresh)

                # Dilate to connect nearby edges (helps with rotated/partial edges)
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE)
                edges = cv2.dilate(edges, kernel_dilate, iterations=self.DILATE_ITERATIONS)

                # Morphological closing to fill gaps
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, self.CLOSE_KERNEL_SIZE)
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=self.CLOSE_ITERATIONS)

                # Find contours
                contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                # Find the first quadrilateral that meets our criteria
                for contour in contours[: self.MAX_CONTOURS_TO_ANALYZE]:
                    area = cv2.contourArea(contour)
                    area_ratio = area / image_area

                    # Filter by area ratio
                    if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                        continue

                    # Try multiple epsilon values for polygon approximation
                    perimeter = cv2.arcLength(contour, True)
                    for epsilon_factor in self.EPSILON_APPROXIMATION_FACTORS:
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
        for threshold_value in self.BRIGHTNESS_THRESHOLDS:
            _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

            # Morphological closing to fill content holes and connect paper regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.BRIGHT_PAPER_KERNEL_SIZE)
            closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=self.CLOSE_ITERATIONS)

            # Find external contours (outermost boundaries only)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Find the first suitable quadrilateral
            for contour in contours[: self.MAX_BRIGHT_CONTOURS_TO_ANALYZE]:
                area = cv2.contourArea(contour)
                area_ratio = area / image_area

                # Paper should be significant portion of image (15-90%)
                # Lower threshold (15%) to catch tightly-cropped flipcharts
                if area_ratio < self.BRIGHT_MIN_AREA_RATIO or area_ratio > self.BRIGHT_MAX_AREA_RATIO:
                    continue

                # Get convex hull first (straightens curved edges)
                hull = cv2.convexHull(contour)

                # Approximate hull to polygon
                perimeter = cv2.arcLength(hull, True)

                # Try different epsilon values to find a quadrilateral
                for epsilon_factor in self.BRIGHT_EPSILON_FACTORS:
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
        if not (self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO):
            return False

        # Check that quad isn't too close to image edges (likely entire image)
        margin = self.QUAD_VALIDATION_MARGIN
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

            # Add generous margin
            margin_w = int(w * self.EDGE_CROP_MARGIN_RATIO)
            margin_h = int(h * self.EDGE_CROP_MARGIN_RATIO)
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
        edges = cv2.Canny(
            gray, self.ROTATION_CANNY_LOW, self.ROTATION_CANNY_HIGH, apertureSize=self.ROTATION_CANNY_APERTURE
        )

        # Hough line detection
        min_line_length = int(min(img.shape[:2]) * self.HOUGH_MIN_LINE_LENGTH_RATIO)
        lines = cv2.HoughLinesP(
            edges,
            rho=self.HOUGH_RHO,
            theta=self.HOUGH_THETA,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=min_line_length,
            maxLineGap=self.HOUGH_MAX_LINE_GAP,
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

        # Only return angle if it's significant
        if abs(median_angle) > self.MIN_ROTATION_ANGLE_DEGREES:
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
