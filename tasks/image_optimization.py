import cv2

from .task_item import FileTask, TaskProcessor
from .registry import TaskRegistry


@TaskRegistry.register("image_optimization")
class ImageOptimizationTask(TaskProcessor):
    """
    Optimizes flipchart images in memory.
    Crops central content more conservatively using adaptive threshold and dilation.
    """

    def __init__(self, padding: int = 10, min_crop_ratio: float = 0.8):
        """
        :param padding: pixels around detected content
        :param min_crop_ratio: minimum fraction of original size to keep
        """
        self.padding = padding
        self.min_crop_ratio = min_crop_ratio

    def process(self, task: FileTask) -> FileTask:
        """
        Process a FileTask by optimizing its image.

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

        h, w = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to highlight text/drawings
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use bounding box of all contours combined for more robust crop
            x1_list, y1_list, x2_list, y2_list = [], [], [], []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                x1_list.append(x)
                y1_list.append(y)
                x2_list.append(x + cw)
                y2_list.append(y + ch)

            x1 = max(min(x1_list) - self.padding, 0)
            y1 = max(min(y1_list) - self.padding, 0)
            x2 = min(max(x2_list) + self.padding, w)
            y2 = min(max(y2_list) + self.padding, h)

            # Avoid over-cropping
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < w * self.min_crop_ratio:
                x1, x2 = 0, w
            if crop_h < h * self.min_crop_ratio:
                y1, y2 = 0, h

            cropped = img[y1:y2, x1:x2]
        else:
            cropped = img  # fallback

        task.img = cropped
        return task
