# Flipchart OCR & PDF Pipeline

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

3. Create a `.env` file with your NVIDIA API key:

   ```
   NVIDIA_API_KEY=your_api_key_here
   ```

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

## License

MIT License © bumbleflies UG
