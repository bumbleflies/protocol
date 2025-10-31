"""Tests for main.py entry point and configuration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import argparse

import pytest
import yaml

import main
from pipeline.config import PipelineConfig


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_logging_creates_handlers(self, tmp_path):
        """Test that setup_logging creates file and console handlers."""
        log_file = tmp_path / "test_debug.log"

        with (
            patch("logging.basicConfig") as mock_basic,
            patch("logging.StreamHandler") as mock_stream,
            patch("logging.getLogger") as mock_get_logger,
        ):

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            main.setup_logging(str(log_file))

            # Verify basicConfig was called
            mock_basic.assert_called_once()
            assert mock_basic.call_args[1]["level"] == 10  # logging.DEBUG

            # Verify StreamHandler was created
            mock_stream.assert_called_once()

            # Verify handler was added to logger
            mock_logger.addHandler.assert_called_once()


class TestMainWithConfig:
    """Test main_with_config function."""

    def test_main_with_config_missing_file(self):
        """Test main_with_config raises error for missing config file."""
        args = argparse.Namespace(config="nonexistent.yaml", input=".", extension=".jpg", output=None, no_ocr=False)

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            main.main_with_config("nonexistent.yaml", args)

    def test_main_with_config_loads_yaml(self, tmp_path):
        """Test main_with_config loads and parses YAML config."""
        # Create minimal config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {}},
            ],
            "input_dir": "test_images",
            "extension": ".png",
            "output_file": "output.pdf",
            "enable_ocr": False,
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        args = argparse.Namespace(config=str(config_file), input=None, extension=".jpg", output=None, no_ocr=False)

        with (
            patch("main.load_dotenv"),
            patch("main.PipelineBuilder") as mock_builder,
            patch("main.FileLoader") as mock_loader,
            patch("main.WorkflowMonitor") as mock_monitor,
            patch("os.getenv") as mock_getenv,
            patch("main._check_input_files") as mock_check_files,
        ):

            # Mock builder
            mock_builder_instance = MagicMock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance.build.return_value = ([], [MagicMock()])

            # Mock loader
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            # Mock monitor
            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance

            # Mock API key
            mock_getenv.return_value = "test_api_key"

            # Mock file checking to simulate files existing
            mock_check_files.return_value = (True, [MagicMock()])  # Directory exists, has files

            main.main_with_config(str(config_file), args)

            # Verify config was loaded
            mock_builder.assert_called_once()

            # Verify FileLoader was created
            mock_loader.assert_called_once()

    def test_main_with_config_cli_overrides(self, tmp_path):
        """Test that CLI args override config file values."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "tasks": [{"name": "optimize", "task_type": "image_optimization", "params": {}}],
            "input_dir": "original_dir",
            "extension": ".jpg",
            "output_file": "original.pdf",
            "enable_ocr": True,
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # CLI args that should override config (use non-default values)
        args = argparse.Namespace(
            config=str(config_file),
            input="override_dir",  # Different from default "."
            extension=".png",  # Different from default ".jpg"
            output="override.pdf",
            no_ocr=False,  # Don't override OCR
        )

        with (
            patch("main.load_dotenv"),
            patch("main.load_config_from_dict") as mock_load_config,
            patch("main.PipelineBuilder") as mock_builder,
            patch("main.FileLoader") as mock_loader,
            patch("main.WorkflowMonitor") as mock_monitor,
            patch("os.getenv") as mock_getenv,
        ):

            mock_builder_instance = MagicMock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance.build.return_value = ([], [MagicMock()])

            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance

            mock_getenv.return_value = "test_api_key"

            # Mock config loading
            mock_config = MagicMock()
            mock_config.input_dir = "override_dir"
            mock_config.extension = ".jpg"
            mock_load_config.return_value = mock_config

            main.main_with_config(str(config_file), args)

            # Verify load_config_from_dict was called with overridden values
            mock_load_config.assert_called_once()
            call_args = mock_load_config.call_args[0][0]

            # Check overrides were applied
            assert call_args["input_dir"] == "override_dir"
            assert call_args["extension"] == ".png"
            assert call_args["output_file"] == "override.pdf"

    def test_main_with_config_output_updates_save_task(self, tmp_path):
        """Test that -o flag updates both output_file and save_pdf task's output_path."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {}},
                {"name": "save", "task_type": "save_pdf", "params": {"output_path": "original.pdf"}},
            ],
            "input_dir": ".",
            "extension": ".jpg",
            "output_file": "original.pdf",
            "enable_ocr": False,
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # CLI args with custom output
        args = argparse.Namespace(
            config=str(config_file), input=None, extension=".jpg", output="custom_output.pdf", no_ocr=False
        )

        with (
            patch("main.load_dotenv"),
            patch("main.load_config_from_dict") as mock_load_config,
            patch("main.PipelineBuilder") as mock_builder,
            patch("main.FileLoader") as mock_loader,
            patch("main.WorkflowMonitor") as mock_monitor,
            patch("main._check_input_files") as mock_check_files,
        ):

            # Mock file checking to simulate files existing
            mock_check_files.return_value = (True, [MagicMock()])

            mock_builder_instance = MagicMock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance.build.return_value = ([], [MagicMock()])

            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance

            # Mock config loading
            mock_config = MagicMock()
            mock_config.input_dir = "."
            mock_config.extension = ".jpg"
            mock_load_config.return_value = mock_config

            main.main_with_config(str(config_file), args)

            # Verify load_config_from_dict was called
            mock_load_config.assert_called_once()
            call_args = mock_load_config.call_args[0][0]

            # Check output_file was updated
            assert call_args["output_file"] == "custom_output.pdf"

            # Check save_pdf task's output_path was also updated
            save_task = next(t for t in call_args["tasks"] if t["task_type"] == "save_pdf")
            assert save_task["params"]["output_path"] == "custom_output.pdf"

    def test_main_with_config_no_ocr_flag(self, tmp_path):
        """Test that --no-ocr flag removes upload and ocr tasks."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {}},
                {"name": "upload", "task_type": "upload", "params": {}},
                {"name": "ocr", "task_type": "ocr", "params": {}},
                {"name": "save", "task_type": "save_pdf", "params": {}},
            ],
            "input_dir": ".",
            "extension": ".jpg",
            "enable_ocr": True,
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        args = argparse.Namespace(config=str(config_file), input=None, extension=".jpg", output=None, no_ocr=True)

        with (
            patch("main.load_dotenv"),
            patch("main.load_config_from_dict") as mock_load_config,
            patch("main.PipelineBuilder") as mock_builder,
            patch("main.FileLoader") as mock_loader,
            patch("main.WorkflowMonitor") as mock_monitor,
        ):

            mock_builder_instance = MagicMock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance.build.return_value = ([], [MagicMock()])

            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance

            mock_config = MagicMock()
            mock_config.input_dir = "."
            mock_config.extension = ".jpg"
            mock_load_config.return_value = mock_config

            main.main_with_config(str(config_file), args)

            # Verify config was modified
            mock_load_config.assert_called_once()
            call_args = mock_load_config.call_args[0][0]

            # Check enable_ocr was set to False
            assert call_args["enable_ocr"] is False

            # Check upload and ocr tasks were removed
            task_names = [t["name"] for t in call_args["tasks"]]
            assert "upload" not in task_names
            assert "ocr" not in task_names
            assert "optimize" in task_names
            assert "save" in task_names

    def test_main_with_config_missing_api_key(self, tmp_path):
        """Test main_with_config logs warning and skips OCR when API key is missing."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {}},
                {"name": "upload", "task_type": "upload", "params": {}},
                {"name": "ocr", "task_type": "ocr", "params": {}},
                {"name": "save", "task_type": "save_pdf", "params": {}},
            ],
            "input_dir": ".",
            "extension": ".jpg",
            "enable_ocr": True,
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        args = argparse.Namespace(config=str(config_file), input=None, extension=".jpg", output=None, no_ocr=False)

        with (
            patch("main.load_dotenv"),
            patch("main.load_config_from_dict") as mock_load_config,
            patch("main.PipelineBuilder") as mock_builder,
            patch("main.FileLoader") as mock_loader,
            patch("main.WorkflowMonitor") as mock_monitor,
            patch("os.getenv") as mock_getenv,
            patch("logging.warning") as mock_warning,
            patch("main._check_input_files") as mock_check_files,
        ):

            mock_builder_instance = MagicMock()
            mock_builder.return_value = mock_builder_instance
            mock_builder_instance.build.return_value = ([], [MagicMock()])

            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance

            # No API key
            mock_getenv.return_value = None

            mock_config = MagicMock()
            mock_config.input_dir = "."
            mock_config.extension = ".jpg"
            mock_load_config.return_value = mock_config

            # Mock file checking to simulate files existing
            mock_check_files.return_value = (True, [MagicMock()])  # Directory exists, has files

            main.main_with_config(str(config_file), args)

            # Verify warning was logged
            mock_warning.assert_called()
            warning_message = str(mock_warning.call_args[0][0])
            assert "NVIDIA_API_KEY" in warning_message
            assert ".env" in warning_message

            # Verify config was modified to skip OCR
            mock_load_config.assert_called_once()
            call_args = mock_load_config.call_args[0][0]
            assert call_args["enable_ocr"] is False
            task_names = [t["name"] for t in call_args["tasks"]]
            assert "upload" not in task_names
            assert "ocr" not in task_names


class TestMain:
    """Test main() entry point."""

    def test_main_parses_args(self):
        """Test that main() parses command line arguments."""
        test_args = [
            "--config",
            "test.yaml",
            "-i",
            "images/",
            "-e",
            ".png",
            "-o",
            "output.pdf",
            "--no-ocr",
        ]

        with (
            patch("sys.argv", ["main.py"] + test_args),
            patch("main.setup_logging"),
            patch("main.main_with_config") as mock_main_config,
        ):

            main.main()

            # Verify main_with_config was called
            mock_main_config.assert_called_once()

            # Check args
            args = mock_main_config.call_args[0][1]
            assert args.config == "test.yaml"
            assert args.input == "images/"
            assert args.extension == ".png"
            assert args.output == "output.pdf"
            assert args.no_ocr is True

    def test_main_default_args(self):
        """Test that main() uses default arguments."""
        with (
            patch("sys.argv", ["main.py"]),
            patch("main.setup_logging"),
            patch("main.main_with_config") as mock_main_config,
        ):

            main.main()

            mock_main_config.assert_called_once()

            args = mock_main_config.call_args[0][1]
            assert args.config == "pipeline_config.yaml"
            assert args.input == "."
            assert args.extension == ".jpg"
            assert args.no_ocr is False

    def test_main_calls_setup_logging(self):
        """Test that main() calls setup_logging."""
        with patch("sys.argv", ["main.py"]), patch("main.setup_logging") as mock_setup, patch("main.main_with_config"):

            main.main()

            # Verify setup_logging was called
            mock_setup.assert_called_once()


class TestIntegration:
    """Integration tests for main workflow."""

    def test_pipeline_execution_no_errors(self, tmp_path):
        """Test that main_with_config executes without errors."""
        # Create minimal config
        config_file = tmp_path / "config.yaml"
        config_data = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {}},
            ],
            "input_dir": ".",
            "extension": ".jpg",
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        args = argparse.Namespace(config=str(config_file), input=".", extension=".jpg", output=None, no_ocr=True)

        # Mock to prevent actual pipeline execution
        with (
            patch("main.load_dotenv"),
            patch("main.FileLoader") as mock_loader,
            patch("main.WorkflowMonitor") as mock_monitor,
            patch("main.PipelineBuilder") as mock_builder,
            patch("main._check_input_files") as mock_check_files,
        ):

            mock_builder_instance = MagicMock()
            mock_builder.return_value = mock_builder_instance

            # Mock workers with proper start/stop/join methods
            mock_workers = [MagicMock() for _ in range(1)]
            mock_queues = [MagicMock() for _ in range(2)]
            mock_builder_instance.build.return_value = (mock_workers, mock_queues)

            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance

            # Mock file checking to simulate files existing
            mock_check_files.return_value = (True, [MagicMock()])  # Directory exists, has files

            # Should execute without errors
            main.main_with_config(str(config_file), args)

            # Verify components were created
            mock_builder.assert_called_once()
            mock_loader.assert_called_once()
            mock_monitor.assert_called_once()

            # Verify workers were started
            for worker in mock_workers:
                worker.start.assert_called_once()

            # Verify monitor was started and stopped
            mock_monitor_instance.start.assert_called_once()
            mock_monitor_instance.stop.assert_called_once()
            mock_monitor_instance.join.assert_called_once()

    def test_pdf_saved_log_message_appears(self, tmp_path, caplog):
        """Test that 'PDF saved successfully' log message appears when PDF is saved."""
        import cv2
        import logging
        import numpy as np
        from tasks.save_pdf import PDFSaveTask
        from tasks.task_item import FileTask, StatusTask

        # Set caplog to capture from tasks.save_pdf logger
        caplog.set_level(logging.INFO, logger="tasks.save_pdf")

        # Create a simple test image in memory
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Gray image

        # Create file tasks with images
        task1 = FileTask(file_path=Path("test1.jpg"), sort_key=1.0)
        task1.img = img
        task2 = FileTask(file_path=Path("test2.jpg"), sort_key=2.0)
        task2.img = img

        # Create PDFSaveTask and process tasks
        output_pdf = tmp_path / "test_output.pdf"
        pdf_task = PDFSaveTask(output_path=str(output_pdf))

        pdf_task.process(task1)
        pdf_task.process(task2)
        pdf_task.finalize()

        # Process StatusTask after finalize (mimics real pipeline flow)
        status_task = StatusTask(files_processed=2)
        pdf_task.process(status_task)

        # Verify PDF was created
        assert output_pdf.exists()

        # Verify success message is in StatusTask
        assert any("PDF saved successfully" in msg for msg in status_task.messages)

    def test_pipeline_completes_with_empty_queues(self, tmp_path):
        """Test that all queues are empty after pipeline completes with sample data."""
        import cv2
        import numpy as np
        import queue
        import time
        from pipeline.builder import PipelineBuilder
        from pipeline.config import load_config_from_dict
        from pipeline.file_loader import FileLoader

        # Create sample test images with actual content to detect
        test_images = []
        for i in range(3):
            img_path = tmp_path / f"0{i + 1}_test.jpg"
            # Create image with white rectangle on black background (easier to detect)
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            img[50:350, 100:500] = [255, 255, 255]  # White rectangle
            cv2.imwrite(str(img_path), img)
            test_images.append(img_path)

        # Create config
        config_dict = {
            "tasks": [
                {"name": "optimize", "task_type": "image_optimization", "params": {"target_width": 800}},
                {"name": "save", "task_type": "save_pdf", "params": {"output_path": str(tmp_path / "output.pdf")}},
            ]
        }
        config = load_config_from_dict(config_dict)

        # Build pipeline
        builder = PipelineBuilder(config)
        workers, queues = builder.build()

        # Start all workers
        for worker in workers:
            worker.start()

        # Load files
        file_loader = FileLoader(target_q=queues[0], input_dir=str(tmp_path), extension=".jpg")
        file_loader.load_files()

        # Wait for processing to complete
        # Wait until all workers have processed FinalizeTask
        max_wait = 30  # 30 seconds max
        start_time = time.time()
        while time.time() - start_time < max_wait:
            all_done = all(w.finalize_done_event.is_set() for w in workers)
            if all_done:
                break
            time.sleep(0.1)

        # Stop workers
        for worker in workers:
            worker.stop(timeout=5)
            worker.join(timeout=5)

        # Verify all input queues (used by workers) are empty
        # Note: builder creates len(tasks)+1 queues, but only len(tasks) are input queues
        for i in range(len(workers)):
            q = queues[i]
            assert q.empty(), f"Queue {i} (input to worker '{workers[i].name}') is not empty, has {q.qsize()} items"

        # The last queue is the output of the last worker
        # It will contain FinalizeTask and StatusTask that pass through since nothing consumes it
        last_worker = workers[-1]
        if last_worker.output_q is not None:
            # Drain the final output queue (expected to have FinalizeTask and StatusTask)
            items_in_last_queue = []
            while not queues[-1].empty():
                items_in_last_queue.append(queues[-1].get())

            # Verify these are the expected sentinel tasks
            from tasks.task_item import FinalizeTask, StatusTask

            assert (
                len(items_in_last_queue) == 2
            ), f"Expected 2 items (FinalizeTask, StatusTask), got {len(items_in_last_queue)}"
            assert isinstance(items_in_last_queue[0], FinalizeTask), "First item should be FinalizeTask"
            assert isinstance(items_in_last_queue[1], StatusTask), "Second item should be StatusTask"

            # Now the queue should be empty
            assert queues[-1].empty(), "Final output queue should be empty after draining"

        # Note: We don't assert on PDF creation here because the image optimization
        # may not produce images suitable for PDF generation (no valid quadrilaterals detected).
        # The key test is that all queues are properly drained.

        # Verify all workers completed their finalize tasks
        for worker in workers:
            assert worker.finalize_done_event.is_set(), f"Worker {worker.name} did not complete finalize"
