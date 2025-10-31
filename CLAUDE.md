# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based flipchart OCR pipeline that processes images of flipcharts, performs OCR using NVIDIA's OCR services, and generates annotated PDFs. The project uses a worker-based queue architecture for asynchronous processing.

## Development Commands

### Running the Pipeline

Basic usage:
```bash
python main.py -i <input_directory> -e <file_extension> -o <output_pdf>
```

Examples:
```bash
# Process all .jpg files in current directory
python main.py

# Process images from specific directory
python main.py -i images/ -o flipcharts.pdf

# Skip OCR and just combine images into PDF
python main.py --no-ocr -i images/ -o output.pdf
```

### Environment Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with:
```
NVIDIA_API_KEY=your_api_key_here
```

## Architecture

### Worker-Queue Pattern

The pipeline uses a threaded worker architecture with queues connecting each processing stage:

```
FileLoader → q_in → Worker(optimize) → q_opt → Worker(upload) → q_upload → Worker(ocr) → q_ocr → Worker(save)
```

- **FileLoader** ([pipeline/file_loader.py](pipeline/file_loader.py)): Scans input directory and loads files into the first queue. Files are sorted by numeric prefix (e.g., `01.1_file.jpg` → sort key 1.1).

- **Worker** ([pipeline/worker.py](pipeline/worker.py)): Generic threaded worker that takes tasks from `input_q`, processes them via `process_fn`, and puts results into `output_q`. Each worker tracks current task and completion count.

- **WorkflowMonitor** ([pipeline/monitor.py](pipeline/monitor.py)): Live monitoring thread that displays pipeline status using Rich library tables. Shows what each worker is processing, queue sizes, and completion counts.

### Task Flow

Tasks flow through the pipeline in two forms:

1. **FileTask** ([tasks/task_item.py](tasks/task_item.py)): Carries image data and metadata through pipeline stages. Contains:
   - `file_path`: Original file location
   - `sort_key`: Numeric sorting key extracted from filename
   - `img`: OpenCV image array (populated after optimization)
   - `asset_id`: NVIDIA upload asset UUID (populated after upload)
   - `ocr_boxes`: List of OCRBox objects (populated after OCR)

2. **FinalizeTask** ([tasks/task_item.py](tasks/task_item.py)): Sentinel task injected by FileLoader after all files. When PDFSaveTask receives this, it triggers final PDF generation.

### Processing Tasks

Each processing stage is implemented as a callable class:

- **ImageOptimizationTask** ([tasks/image_optimization.py](tasks/image_optimization.py)): Loads image, applies adaptive thresholding and dilation to detect content, crops with padding while avoiding over-cropping (min_crop_ratio=0.8).

- **UploadTask** ([tasks/ocr.py](tasks/ocr.py:61-78)): Encodes optimized image as JPEG and uploads to NVIDIA NVCF asset storage. Stores asset_id in FileTask.

- **OCRTask** ([tasks/ocr.py](tasks/ocr.py:81-163)): Calls NVIDIA OCR API with asset_id, downloads ZIP response, extracts JSON with polygon coordinates and text labels, populates ocr_boxes sorted top-down, left-right.

- **PDFSaveTask** ([tasks/save_pdf.py](tasks/save_pdf.py)): Accumulates FileTask objects until FinalizeTask arrives. Then:
  1. Sorts collected tasks by sort_key
  2. Converts OpenCV images to PIL and creates temporary PDF
  3. Reads PDF with PyPDF2 and adds text annotations at OCR box coordinates
  4. Saves annotated PDF (filters boxes with confidence < 0.5)

### Command-Line Arguments

Defined in [main.py](main.py:32-46):
- `-i/--input`: Input directory (default: current directory)
- `-e/--extension`: File extension to process (default: `.jpg`)
- `-o/--output`: Output PDF filename (default: `combined-YYYYMMDD-HHMMSS.pdf`)
- `--no-ocr`: Skip upload/OCR stages and just combine images

### Logging

Dual logging setup ([main.py](main.py:15-26)):
- DEBUG level logs to `debug.log` file
- INFO level logs to console
- All log messages include timestamp, level, and thread name

## Key Implementation Details

### File Sorting

FileLoader extracts numeric prefixes from filenames using regex `^(\d+)(?:\.(\d+))?_?`:
- `01.1_file.jpg` → sort key 1.1
- `02_file.jpg` → sort key 2.0
- Files without numeric prefix → sort key 0.0

### NVIDIA OCR Integration

OCR process ([tasks/ocr.py](tasks/ocr.py)):
1. Upload image bytes to NVCF asset storage
2. POST to `https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet` with asset_id
3. Response is ZIP containing `.response` JSON file
4. JSON contains metadata array with label (text) and polygon (x1-4, y1-4) coordinates

### PDF Annotation

PyPDF2 annotations ([tasks/save_pdf.py](tasks/save_pdf.py:66-120)):
- Coordinate system requires Y-axis flip: `yLL = page_height - max(y_coords)`
- Uses `AnnotationBuilder.text()` to create sticky note annotations
- Annotations are collapsed by default (`open=False`)
- Low confidence boxes (< 0.5) are skipped

### Thread Coordination

Workers use `finalize_done_event` to coordinate shutdown:
- Worker processes FinalizeTask and sets event
- `worker.stop()` waits for this event before setting stop signal
- Ensures all work completes before threads exit
