# Flipchart OCR & PDF Pipeline

[![CI](https://github.com/bumbleflies/protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/bumbleflies/protocol/actions/workflows/ci.yml)
[![Release](https://github.com/bumbleflies/protocol/actions/workflows/release.yml/badge.svg)](https://github.com/bumbleflies/protocol/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/bumbleflies/protocol/branch/main/graph/badge.svg)](https://codecov.io/gh/bumbleflies/protocol)
[![CodeQL](https://github.com/bumbleflies/protocol/actions/workflows/codeql.yml/badge.svg)](https://github.com/bumbleflies/protocol/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-based pipeline for processing images of flipcharts, performing OCR (Optical Character Recognition) using NVIDIA
OCR services, optionally uploading to NVCF, annotating OCR results, and generating a combined PDF. Designed for quick
processing, annotation, and archival of flipchart content.

---

## Features

* Load images from a directory (supports `.jpg` or custom extensions).
* Image preprocessing and optimization.
* Upload images to NVIDIA NVCF (optional).
* OCR detection with German language support.
* Annotate PDF with OCR results (bounding boxes and text).
* Combine multiple images into a single PDF.
* Command-line interface with flexible input/output options.
* Workflow monitoring with worker threads for asynchronous processing.

---

## Requirements

* Python 3.11+
* Libraries:

  * `opencv-python`
  * `numpy`
  * `requests`
  * `python-dotenv`
  * `PyPDF2>=3.0.0`
  * `reportlab`
  * `Pillow`
* NVIDIA OCR API key

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bumbleflies/protocol.git
   cd flipchart-pipeline
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your NVIDIA API key (optional for OCR):

   ```bash
   cp .env.example .env
   # Edit .env and add your NVIDIA API key
   ```

   Get your API key from: https://build.nvidia.com/explore/discover

   **Note:** If no API key is provided, the pipeline will automatically skip OCR and just combine images into a PDF.

---

## Usage

Run the pipeline from the command line:

```bash
python main.py -i /path/to/images -e .jpg -o output.pdf
```

### Options:

| Flag                | Description                       | Default                        |
|---------------------|-----------------------------------|--------------------------------|
| `-i`, `--input`     | Input directory containing images | `.` (current directory)        |
| `-e`, `--extension` | File extension to process         | `.jpg`                         |
| `-o`, `--output`    | Output PDF filename               | `combined-YYYYMMDD-HHMMSS.pdf` |
| `--no-ocr`          | Skip OCR step (images only)       | `False`                        |
| `--config`          | Path to YAML configuration file   | `pipeline_config.yaml`         |

---

## Example Workflow

1. Place flipchart images in a folder (`images/`).
2. Run the pipeline:

   ```bash
   python main.py -i images/ -o flipcharts.pdf
   ```
3. The pipeline will:

  * Optimize images
  * Upload (if configured)
  * Run OCR
  * Annotate detected text
  * Produce `flipcharts.pdf` with OCR annotations

---

## Project Structure

```
flipchart-pipeline/
├─ main.py                # Entry point
├─ pipeline/              # Core workflow classes
│  ├─ file_loader.py
│  ├─ worker.py
│  └─ monitor.py
├─ tasks/                 # Task modules
│  ├─ image_optimization.py
│  ├─ upload_task.py
│  ├─ ocr_task.py
│  └─ save_pdf.py
├─ .env                   # Environment variables (API keys)
├─ requirements.txt
└─ README.md
```

---

## Notes

* All sensitive information (API keys) should be stored in `.env`.
* OCR language is set to German by default; this can be configured in `OCRTask`.
* Worker threads allow asynchronous image processing for large datasets.

---

## Recent Updates (SOLID Refactoring)

The codebase has been refactored to follow SOLID principles! 🎉

**Key improvements:**
- ✅ Abstract base classes for task processors (LSP compliance)
- ✅ Provider abstraction for OCR services (DIP compliance)
- ✅ Task registry system (OCP compliance)
- ✅ Configuration-based pipeline with dependency injection
- ✅ 40 comprehensive tests with 50% code coverage
- ✅ Backward compatible with existing usage

**New features:**
```bash
# Use YAML configuration for flexible pipelines
python main.py --config pipeline_config.yaml
```

**Documentation:**
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Complete refactoring details
- [CLAUDE.md](CLAUDE.md) - Architecture documentation
- [TEST_RESULTS.md](TEST_RESULTS.md) - Test coverage and results
- [SEMANTIC_RELEASE.md](SEMANTIC_RELEASE.md) - Automated versioning and releases
- [tests/README.md](tests/README.md) - Testing guide

**Running tests:**
```bash
pip install -r requirements.txt
pip install -r test_requirements.txt
pytest tests/ -v
```

**Releases:**
This project uses [Python Semantic Release](https://python-semantic-release.readthedocs.io/) for automated versioning.
Use [Conventional Commits](https://www.conventionalcommits.org/) for automatic version bumps:
- `feat:` → Minor version bump (0.x.0)
- `fix:` → Patch version bump (0.0.x)
- `feat!:` or `BREAKING CHANGE:` → Major version bump (x.0.0)

See [SEMANTIC_RELEASE.md](SEMANTIC_RELEASE.md) for detailed instructions.

---

## License

MIT License © bumbleflies UG
