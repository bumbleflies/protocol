import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from pipeline import FileLoader
from pipeline.monitor import WorkflowMonitor
from pipeline.builder import PipelineBuilder
from pipeline.config import load_config_from_dict
from tasks.nvidia_ocr_provider import NvidiaAssetUploader, NvidiaOCRProvider


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


def main_with_config(config_path: str, args) -> None:
    """
    Configuration-based pipeline implementation.
    Uses YAML config files and dependency injection.
    """
    load_dotenv()

    # Load configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.input != ".":
        config_data["input_dir"] = args.input
    if args.extension != ".jpg":
        config_data["extension"] = args.extension
    if args.output:
        config_data["output_file"] = args.output
    if args.no_ocr:
        config_data["enable_ocr"] = False
        # Remove upload and ocr tasks
        config_data["tasks"] = [t for t in config_data["tasks"] if t["name"] not in ["upload", "ocr"]]

    # Check for API key and auto-disable OCR if missing
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key and config_data.get("enable_ocr", True):
        # No API key - automatically skip OCR tasks
        logging.warning(
            "NVIDIA_API_KEY not set. Skipping OCR tasks.\n"
            "To enable OCR, create a .env file with: NVIDIA_API_KEY=your_api_key_here\n"
            "Get your API key from: https://build.nvidia.com/explore/discover"
        )
        config_data["enable_ocr"] = False
        config_data["tasks"] = [t for t in config_data["tasks"] if t["name"] not in ["upload", "ocr"]]

    config = load_config_from_dict(config_data)

    # Build pipeline with dependency injection
    builder = PipelineBuilder(config)

    # Register providers if API key is available
    if api_key:
        builder.register_provider("nvidia_uploader", NvidiaAssetUploader(api_key))
        builder.register_provider("nvidia_ocr", NvidiaOCRProvider(api_key))

    # Check if input directory has files before starting pipeline
    input_path = Path(config.input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {config.input_dir}")
        return

    files = list(input_path.glob(f"*{config.extension}"))
    if not files:
        logging.warning(f"No files with extension '{config.extension}' found in {config.input_dir}")
        logging.info("Pipeline not started - no files to process")
        return

    logging.info(f"Found {len(files)} file(s) to process")

    # Build workers and queues
    workers, queues = builder.build()

    # Start all workers
    for w in workers:
        w.start()

    # Load files
    file_loader = FileLoader(input_dir=config.input_dir, extension=config.extension, target_q=queues[0])
    file_loader.load_files()

    # Start monitor (non-daemon to ensure proper cleanup)
    monitor = WorkflowMonitor(workers)
    monitor.start()

    # Stop all workers and wait for completion
    for w in workers:
        w.stop()

    # Stop monitor and wait for it to finish
    monitor.stop()
    monitor.join(timeout=5)  # Wait up to 5 seconds for monitor to finish


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Flipchart pipeline: optimize images and create PDF")
    parser.add_argument("-i", "--input", type=str, default=".", help="Input directory containing images")
    parser.add_argument("-e", "--extension", type=str, default=".jpg", help="File extension to process")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"combined-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf",
        help="Output PDF filename",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR step and use only images for PDF")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to YAML configuration file (default: pipeline_config.yaml)",
    )
    args = parser.parse_args()

    logging.info(f"Using config-based pipeline: {args.config}")
    main_with_config(args.config, args)


if __name__ == "__main__":
    main()
