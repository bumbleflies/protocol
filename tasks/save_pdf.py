import logging
from pathlib import Path
from typing import List

import cv2
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
from pypdf.annotations import Text

from .task_item import FileTask, FinalizableTaskProcessor
from .registry import TaskRegistry

logger = logging.getLogger(__name__)


@TaskRegistry.register("save_pdf")
class PDFSaveTask(FinalizableTaskProcessor):
    def __init__(self, output_path: str = "combined.pdf"):
        self.output_path = Path(output_path)
        self.collected_tasks: List[FileTask] = []

    def process(self, task: FileTask) -> FileTask:
        """
        Process a FileTask by collecting it for PDF generation.

        Args:
            task: The FileTask to add to the collection

        Returns:
            The same FileTask (unmodified)
        """
        if task.img is not None:
            self.collected_tasks.append(task)
            logger.debug(f"FileTask stored for PDF: {task.file_path}")
        return task

    def finalize(self) -> None:
        """
        Generate the final PDF when FinalizeTask is received.
        This method is called by Worker when the pipeline completes.
        """
        logger.debug("Finalize signal received, generating annotated PDF...")
        self._finalize_pdf()
        logger.debug("PDF generation completed")

    def _finalize_pdf(self) -> None:
        """Combine collected images into a PDF and add annotations."""
        if not self.collected_tasks:
            logger.warning("No images to save in PDF.")
            return

        sorted_tasks = sorted(self.collected_tasks, key=lambda t: t.sort_key)
        logger.debug(f"Sorting {len(sorted_tasks)} images by sort_key")

        # Save temporary PDF with images only
        temp_pdf_path = self.output_path.with_suffix(".temp.pdf")
        pil_images = []
        for t in sorted_tasks:
            if t.img is None:
                logger.warning(f"Skipping task with missing image: {t.file_path}")
                continue
            img_rgb = cv2.cvtColor(t.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_images.append(pil_img)

        if pil_images:
            first_image, *rest = pil_images
            first_image.save(temp_pdf_path, save_all=True, append_images=rest)
            logger.debug(f"Temporary PDF saved to {temp_pdf_path}")
        else:
            logger.warning("No valid images found to save in PDF.")
            return

        # Add annotations with pypdf
        self._add_annotations(temp_pdf_path, self.output_path, sorted_tasks)
        temp_pdf_path.unlink(missing_ok=True)
        logger.info(f"PDF saved successfully: {self.output_path.absolute()}")

    def _add_annotations(self, pdf_path: Path, output_path: Path, tasks: List[FileTask]) -> None:
        """Add static text annotations (like sticky notes) to the PDF."""
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()

        for page_idx, page in enumerate(reader.pages):
            writer.add_page(page)
            page_height = page.mediabox.top

            if page_idx >= len(tasks):
                continue

            task = tasks[page_idx]
            if not getattr(task, "ocr_boxes", None):
                continue

            for box in task.ocr_boxes:
                if box.confidence < 0.5:
                    logger.debug(f"Skipping low confidence annotation {box}")
                    continue  # skip low-confidence boxes
                # Calculate bounding rectangle
                x_coords = [box.x1, box.x2, box.x3, box.x4]
                y_coords = [box.y1, box.y2, box.y3, box.y4]

                xLL = min(x_coords)
                yLL = page_height - max(y_coords)  # flip y
                xUR = max(x_coords)
                yUR = page_height - min(y_coords)

                rect = RectangleObject((xLL, yLL, xUR, yUR))

                try:
                    logger.debug(f"Adding annotation {box}")
                    # Add static text annotation
                    annotation = Text(
                        rect=rect,
                        text=box.label or "",
                        open=False,  # collapsed by default
                    )
                    writer.add_annotation(page_number=page_idx, annotation=annotation)

                except Exception as e:
                    logger.warning(f"Failed to add annotation for {task.file_path}: {e}")

        # Save annotated PDF
        with open(output_path, "wb") as f:
            writer.write(f)
