# Test Suite

This directory contains tests for the refactored protocol pipeline, focusing on SOLID principles compliance.

## Running Tests

```bash
# Install production dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r test_requirements.txt

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_task_processor.py -v

# Run with coverage
pytest tests/ --cov=tasks --cov=pipeline -v
```

## Test Files

### test_task_processor.py
Tests for `TaskProcessor` and `FinalizableTaskProcessor` base classes.
- **LSP Compliance**: Verifies Liskov Substitution Principle
- Tests that `FinalizableTaskProcessor` can be used wherever `TaskProcessor` is expected
- Tests proper separation of `process()` and `finalize()` methods

### test_registry.py
Tests for `TaskRegistry` and task discovery.
- **OCP Compliance**: Verifies Open/Closed Principle
- Tests that new tasks can be added without modifying existing code
- Tests registry lookup and error handling

### test_ocr_provider.py
Tests for OCR provider abstraction (`AssetUploader` and `OCRProvider`).
- **DIP Compliance**: Verifies Dependency Inversion Principle
- Tests that high-level code depends on abstractions, not concrete implementations
- Tests provider swapping and mock implementations

### test_pipeline_builder.py
Tests for `PipelineBuilder` and configuration system.
- Tests dependency injection
- Tests configuration loading
- Tests pipeline construction from config

### test_worker.py
Tests for `Worker` integration with `TaskProcessor`.
- Tests worker processing of FileTasks
- Tests FinalizeTask handling
- Tests error handling and timeout behavior

## Test Coverage Goals

- **LSP (Liskov Substitution Principle)**: ✅ Tested in `test_task_processor.py`
- **OCP (Open/Closed Principle)**: ✅ Tested in `test_registry.py`
- **DIP (Dependency Inversion Principle)**: ✅ Tested in `test_ocr_provider.py`
- **Integration**: ✅ Tested in `test_worker.py` and `test_pipeline_builder.py`

## Mock Implementations

The tests include mock implementations that demonstrate the power of the new architecture:

- `MockAssetUploader`: Mock uploader for testing without real API calls
- `MockOCRProvider`: Mock OCR for testing without real NVIDIA API
- `CountingProcessor`: Simple processor for testing worker behavior
- `FinalizableCountingProcessor`: Finalizable processor for testing finalization logic

These mocks can be used as templates for implementing alternative providers (e.g., Google Vision, AWS Textract).
