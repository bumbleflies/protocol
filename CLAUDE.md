# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based flipchart OCR pipeline that processes images of flipcharts, performs OCR using NVIDIA's OCR services, and generates annotated PDFs. The project uses a worker-based queue architecture for asynchronous processing.

**Architecture**: The codebase follows SOLID principles with:
- Abstract base classes for task processors (LSP compliance)
- Provider abstraction for OCR services (DIP compliance)
- Task registry system (OCP compliance)
- Configuration-based pipeline builder with dependency injection

## Development Commands

### Running the Pipeline

The pipeline uses YAML configuration files for setup:

```bash
python main.py [--config pipeline_config.yaml] [options]
```

Examples:
```bash
# Use default config (pipeline_config.yaml)
python main.py

# Use custom config file
python main.py --config my_config.yaml

# Override config with CLI args
python main.py --config pipeline_config.yaml -i images/ -o custom.pdf

# Skip OCR (only combine images to PDF)
python main.py --no-ocr
```

### Running Tests

```bash
# Install dependencies (production + test + dev tools)
pip install -e ".[test,dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=tasks --cov=pipeline -v
```

### Environment Setup

1. Install dependencies:
```bash
# Install in editable mode with all optional dependencies
pip install -e ".[test,dev]"
```

2. Create `.env` file with:
```
NVIDIA_API_KEY=your_api_key_here
```

**Dependency Management**: This project uses `pyproject.toml` (PEP 518/621) as the single source of truth for all dependencies and tool configurations. Optional dependency groups:
- `[test]`: pytest, pytest-cov, pytest-mock, pytest-asyncio
- `[dev]`: flake8, black, mypy (code quality tools)

## Architecture

### SOLID Principles Implementation

The codebase was refactored to address SOLID violations:

1. **Single Responsibility (SRP)**: ✅ Each class has one well-defined responsibility
2. **Open/Closed (OCP)**: ✅ New tasks can be added via registry without modifying existing code
3. **Liskov Substitution (LSP)**: ✅ `FinalizableTaskProcessor` is a proper subtype of `TaskProcessor`
4. **Interface Segregation (ISP)**: ✅ Minimal interfaces for task processors
5. **Dependency Inversion (DIP)**: ✅ High-level code depends on abstractions (`OCRProvider`, `AssetUploader`)

### Task Processor Architecture (LSP Fix)

**Base Classes** ([tasks/task_item.py](tasks/task_item.py)):

```python
class TaskProcessor(ABC):
    """Abstract base for all task processors."""
    @abstractmethod
    def process(self, task: FileTask) -> FileTask:
        pass

class FinalizableTaskProcessor(TaskProcessor):
    """Base for processors that need finalization logic."""
    @abstractmethod
    def finalize(self) -> None:
        pass
```

**Key Improvement**: Task processors no longer need `isinstance(task, FinalizeTask)` checks. The Worker handles FinalizeTask and calls `finalize()` on processors that support it.

**All Tasks Implement**:
- `ImageOptimizationTask` → `TaskProcessor`
- `UploadTask` → `TaskProcessor`
- `OCRTask` → `TaskProcessor`
- `PDFSaveTask` → `FinalizableTaskProcessor` (only one that needs finalization)

### Provider Abstraction (DIP Fix)

**Abstract Interfaces** ([tasks/ocr_provider.py](tasks/ocr_provider.py)):

```python
class AssetUploader(ABC):
    @abstractmethod
    def upload(self, image_bytes: bytes, description: str) -> UUID:
        pass

class OCRProvider(ABC):
    @abstractmethod
    def detect_text(self, asset_id: UUID, image: np.ndarray) -> List[OCRBox]:
        pass
```

**NVIDIA Implementation** ([tasks/nvidia_ocr_provider.py](tasks/nvidia_ocr_provider.py)):
- `NvidiaAssetUploader`: Uploads to NVIDIA NVCF
- `NvidiaOCRProvider`: Calls NVIDIA OCDRNet API

**Benefits**:
- Can swap providers (e.g., Google Vision, AWS Textract)
- Easy to create mock providers for testing
- API keys and URLs isolated in implementations

### Task Registry (OCP Fix)

**Registry System** ([tasks/registry.py](tasks/registry.py)):

```python
@TaskRegistry.register("image_optimization")
class ImageOptimizationTask(TaskProcessor):
    ...
```

**Benefits**:
- New tasks discovered at runtime
- No need to modify imports in main.py
- Third-party tasks can register themselves

### Configuration & Dependency Injection

**Pipeline Builder** ([pipeline/builder.py](pipeline/builder.py)):

```python
builder = PipelineBuilder(config)
builder.register_provider("nvidia_uploader", NvidiaAssetUploader(api_key))
builder.register_provider("nvidia_ocr", NvidiaOCRProvider(api_key))
workers, queues = builder.build()
```

**YAML Config** ([pipeline_config.yaml](pipeline_config.yaml)):

```yaml
tasks:
  - name: upload
    task_type: upload
    params:
      uploader: "@nvidia_uploader"  # Injected provider reference
```

**Benefits**:
- Pipeline configured via YAML, not hardcoded
- Dependency injection makes testing trivial
- Can create different pipelines for different use cases

### Worker-Queue Pattern

The pipeline uses a threaded worker architecture with queues connecting each processing stage:

```
FileLoader → q_in → Worker(optimize) → q_opt → Worker(upload) → q_upload → Worker(ocr) → q_ocr → Worker(save)
```

**Components**:

- **FileLoader** ([pipeline/file_loader.py](pipeline/file_loader.py)): Scans input directory and loads files into the first queue. Files are sorted by numeric prefix (e.g., `01.1_file.jpg` → sort key 1.1).

- **Worker** ([pipeline/worker.py](pipeline/worker.py)): Generic threaded worker that:
  - Takes tasks from `input_q`
  - Calls `processor.process(task)` for FileTask
  - Calls `processor.finalize()` for FinalizeTask (if processor supports it)
  - Puts results into `output_q`
  - Tracks current task and completion count
  - Now includes timeout on `stop()` to prevent hangs

- **WorkflowMonitor** ([pipeline/monitor.py](pipeline/monitor.py)): Live monitoring thread that displays pipeline status using Rich library tables. Shows what each worker is processing, queue sizes, and completion counts.

### Task Flow

Tasks flow through the pipeline in two forms:

1. **FileTask** ([tasks/task_item.py](tasks/task_item.py)): Carries image data and metadata through pipeline stages. Contains:
   - `file_path`: Original file location
   - `sort_key`: Numeric sorting key extracted from filename
   - `img`: OpenCV image array (populated after optimization)
   - `asset_id`: NVIDIA upload asset UUID (populated after upload)
   - `ocr_boxes`: List of OCRBox objects (populated after OCR)

2. **FinalizeTask** ([tasks/task_item.py](tasks/task_item.py)): Sentinel task injected by FileLoader after all files. Worker detects this and calls `finalize()` on processors that support it.

### Processing Tasks

Each processing stage is implemented as a TaskProcessor:

- **ImageOptimizationTask** ([tasks/image_optimization.py](tasks/image_optimization.py)):
  - Implements `TaskProcessor.process()`
  - Loads image, applies adaptive thresholding and dilation
  - Crops with padding while avoiding over-cropping (min_crop_ratio=0.8)

- **UploadTask** ([tasks/ocr.py](tasks/ocr.py)):
  - Implements `TaskProcessor.process()`
  - Takes `AssetUploader` provider via constructor (dependency injection)
  - Encodes optimized image as JPEG and uploads
  - Stores asset_id in FileTask

- **OCRTask** ([tasks/ocr.py](tasks/ocr.py)):
  - Implements `TaskProcessor.process()`
  - Takes `OCRProvider` via constructor (dependency injection)
  - Calls provider's `detect_text()` method
  - Populates ocr_boxes sorted top-down, left-right

- **PDFSaveTask** ([tasks/save_pdf.py](tasks/save_pdf.py)):
  - Implements `FinalizableTaskProcessor.process()` and `finalize()`
  - Accumulates FileTask objects in `process()`
  - Generates PDF in `finalize()` when pipeline completes:
    1. Sorts collected tasks by sort_key
    2. Converts OpenCV images to PIL and creates temporary PDF
    3. Reads PDF with PyPDF2 and adds text annotations at OCR box coordinates
    4. Saves annotated PDF (filters boxes with confidence < 0.5)

### Command-Line Arguments

Defined in [main.py](main.py):
- `-i/--input`: Input directory (default: current directory)
- `-e/--extension`: File extension to process (default: `.jpg`)
- `-o/--output`: Output PDF filename (default: `combined-YYYYMMDD-HHMMSS.pdf`)
- `--no-ocr`: Skip upload/OCR stages and just combine images
- `--config`: Path to YAML configuration file (default: `pipeline_config.yaml`)

### Logging

Dual logging setup ([main.py](main.py)):
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

OCR process ([tasks/nvidia_ocr_provider.py](tasks/nvidia_ocr_provider.py)):
1. Upload image bytes to NVCF asset storage
2. POST to `https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet` with asset_id
3. Response is ZIP containing `.response` JSON file
4. JSON contains metadata array with label (text) and polygon (x1-4, y1-4) coordinates
5. **Platform fix**: Uses `tempfile.gettempdir()` instead of hardcoded `/tmp/`
6. **Resource cleanup**: Removes temp files after processing

### PDF Annotation

PyPDF2 annotations ([tasks/save_pdf.py](tasks/save_pdf.py)):
- Coordinate system requires Y-axis flip: `yLL = page_height - max(y_coords)`
- Uses `AnnotationBuilder.text()` to create sticky note annotations
- Annotations are collapsed by default (`open=False`)
- Low confidence boxes (< 0.5) are skipped

### Thread Coordination

Workers use `finalize_done_event` to coordinate shutdown:
- Worker processes FinalizeTask and sets event
- `worker.stop(timeout)` waits for this event before setting stop signal
- **Safety improvement**: Now includes timeout (default 30s) to prevent infinite hangs
- Ensures all work completes before threads exit

## Testing

**Test Files** ([tests/](tests/)):
- `test_task_processor.py`: LSP compliance tests
- `test_registry.py`: OCP compliance tests
- `test_ocr_provider.py`: DIP compliance tests
- `test_pipeline_builder.py`: Configuration and dependency injection tests
- `test_worker.py`: Worker integration tests

**Running Tests**:
```bash
pytest tests/ -v
pytest tests/ --cov=tasks --cov=pipeline
```

## Adding New Features

### Adding a New Task

1. Create task class extending `TaskProcessor`:
```python
@TaskRegistry.register("my_task")
class MyTask(TaskProcessor):
    def process(self, task: FileTask) -> FileTask:
        # Your logic here
        return task
```

2. Add to config YAML:
```yaml
tasks:
  - name: my_step
    task_type: my_task
    params: {}
```

### Adding a New OCR Provider

1. Implement provider interfaces:
```python
class MyOCRProvider(OCRProvider):
    def detect_text(self, asset_id: UUID, image: np.ndarray) -> List[OCRBox]:
        # Your provider logic
        pass
```

2. Register and inject:
```python
builder.register_provider("my_ocr", MyOCRProvider())
```

3. Reference in config:
```yaml
params:
  provider: "@my_ocr"
```

This architecture makes extending the pipeline straightforward without modifying existing code.
