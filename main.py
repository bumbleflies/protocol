import argparse
import logging
import queue
from datetime import datetime

from dotenv import load_dotenv

from pipeline import FileLoader
from pipeline import Worker
from pipeline.monitor import WorkflowMonitor
from tasks import ImageOptimizationTask, UploadTask, OCRTask
from tasks import PDFSaveTask


def setup_logging(log_file: str = "debug.log") -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logging.getLogger("").addHandler(console)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Flipchart pipeline: optimize images and create PDF")
    parser.add_argument("-i", "--input", type=str, default=".", help="Input directory containing images")
    parser.add_argument("-e", "--extension", type=str, default=".jpg", help="File extension to process")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=f"combined-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf",
        help="Output PDF filename"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR step and use only images for PDF"
    )
    args = parser.parse_args()

    # Queues
    q_in = queue.Queue()
    q_opt = queue.Queue()
    q_upload = queue.Queue()
    q_ocr = queue.Queue()

    # Workers
    workers = [Worker("optimize", q_in, q_opt, ImageOptimizationTask())]

    if not args.no_ocr:
        upload_worker = Worker("upload", q_opt, q_upload, UploadTask())
        ocr_worker = Worker("ocr", q_upload, q_ocr, OCRTask())
        save_worker = Worker("save", q_ocr, None, PDFSaveTask(args.output))
        workers.extend([upload_worker, ocr_worker, save_worker])
    else:
        workers.append(Worker("save", q_opt, None, PDFSaveTask(args.output)))

    # Start all workers
    for w in workers:
        w.start()

    # Load files
    file_loader = FileLoader(input_dir=args.input, extension=args.extension, target_q=q_in)
    file_loader.load_files()

    # Start monitor
    monitor = WorkflowMonitor(workers)
    monitor.start()

    # Stop all workers
    for w in workers:
        w.stop()
    monitor.stop()


if __name__ == "__main__":
    main()
