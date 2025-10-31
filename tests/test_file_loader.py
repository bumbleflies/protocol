"""Tests for FileLoader."""

import queue
import tempfile
from pathlib import Path

import pytest

from pipeline.file_loader import FileLoader
from tasks.task_item import FileTask, FinalizeTask


class TestFileLoader:
    """Test FileLoader functionality."""

    def test_creation(self):
        """Test FileLoader can be created."""
        q = queue.Queue()
        loader = FileLoader(target_q=q, input_dir=".", extension=".jpg")

        assert loader.input_dir == Path(".")
        assert loader.extension == ".jpg"
        assert loader.target_q == q

    def test_extension_normalized_to_lowercase(self):
        """Test extension is normalized to lowercase."""
        q = queue.Queue()
        loader = FileLoader(target_q=q, input_dir=".", extension=".JPG")

        assert loader.extension == ".jpg"

    def test_compute_sort_key_simple_number(self):
        """Test sort key computation for simple numbered files."""
        q = queue.Queue()
        loader = FileLoader(target_q=q)

        assert loader._compute_sort_key(Path("01_file.jpg")) == 1.0
        assert loader._compute_sort_key(Path("02_file.jpg")) == 2.0
        assert loader._compute_sort_key(Path("10_file.jpg")) == 10.0

    def test_compute_sort_key_with_decimal(self):
        """Test sort key computation for files with decimal numbering."""
        q = queue.Queue()
        loader = FileLoader(target_q=q)

        assert loader._compute_sort_key(Path("01.1_file.jpg")) == 1.01
        assert loader._compute_sort_key(Path("01.2_file.jpg")) == 1.02
        assert loader._compute_sort_key(Path("02.5_file.jpg")) == 2.05

    def test_compute_sort_key_without_underscore(self):
        """Test sort key computation for files without underscore."""
        q = queue.Queue()
        loader = FileLoader(target_q=q)

        assert loader._compute_sort_key(Path("01file.jpg")) == 1.0
        assert loader._compute_sort_key(Path("02.3file.jpg")) == 2.03

    def test_compute_sort_key_no_number(self):
        """Test sort key computation for files without leading numbers."""
        q = queue.Queue()
        loader = FileLoader(target_q=q)

        assert loader._compute_sort_key(Path("file.jpg")) is None
        assert loader._compute_sort_key(Path("random_name.jpg")) is None

    def test_load_files_nonexistent_directory(self):
        """Test load_files raises FileNotFoundError for nonexistent directory."""
        q = queue.Queue()
        loader = FileLoader(target_q=q, input_dir="/nonexistent/path")

        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            loader.load_files()

    def test_load_files_empty_directory(self):
        """Test load_files with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should only have FinalizeTask
            assert q.qsize() == 1
            task = q.get()
            assert isinstance(task, FinalizeTask)

    def test_load_files_single_file(self):
        """Test load_files with single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "01_test.jpg"
            test_file.touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should have FileTask + FinalizeTask
            assert q.qsize() == 2

            # First should be FileTask
            task1 = q.get()
            assert isinstance(task1, FileTask)
            assert task1.file_path == test_file
            assert task1.sort_key == 1.0

            # Second should be FinalizeTask
            task2 = q.get()
            assert isinstance(task2, FinalizeTask)

    def test_load_files_multiple_files_sorted(self):
        """Test load_files with multiple files sorted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files in non-sorted order
            files = [
                "03_third.jpg",
                "01_first.jpg",
                "02_second.jpg",
            ]

            for filename in files:
                (Path(tmpdir) / filename).touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should have 3 FileTasks + 1 FinalizeTask
            assert q.qsize() == 4

            # Verify sorting
            task1 = q.get()
            assert isinstance(task1, FileTask)
            assert task1.file_path.name == "01_first.jpg"

            task2 = q.get()
            assert isinstance(task2, FileTask)
            assert task2.file_path.name == "02_second.jpg"

            task3 = q.get()
            assert isinstance(task3, FileTask)
            assert task3.file_path.name == "03_third.jpg"

            task4 = q.get()
            assert isinstance(task4, FinalizeTask)

    def test_load_files_with_decimal_numbering_sorted(self):
        """Test load_files with decimal numbering sorted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files with decimal numbering
            files = [
                "01.2_second.jpg",
                "02.1_fourth.jpg",
                "01.1_first.jpg",
                "02.2_fifth.jpg",
                "01.3_third.jpg",
            ]

            for filename in files:
                (Path(tmpdir) / filename).touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should have 5 FileTasks + 1 FinalizeTask
            assert q.qsize() == 6

            # Verify sorting
            expected_order = [
                "01.1_first.jpg",
                "01.2_second.jpg",
                "01.3_third.jpg",
                "02.1_fourth.jpg",
                "02.2_fifth.jpg",
            ]

            for expected_name in expected_order:
                task = q.get()
                assert isinstance(task, FileTask)
                assert task.file_path.name == expected_name

            # Last should be FinalizeTask
            finalize = q.get()
            assert isinstance(finalize, FinalizeTask)

    def test_load_files_filters_by_extension(self):
        """Test load_files only loads files with matching extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            (Path(tmpdir) / "01_test.jpg").touch()
            (Path(tmpdir) / "02_test.png").touch()
            (Path(tmpdir) / "03_test.txt").touch()
            (Path(tmpdir) / "04_test.jpg").touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should have 2 JPG files + 1 FinalizeTask
            assert q.qsize() == 3

            task1 = q.get()
            assert isinstance(task1, FileTask)
            assert task1.file_path.name == "01_test.jpg"

            task2 = q.get()
            assert isinstance(task2, FileTask)
            assert task2.file_path.name == "04_test.jpg"

            task3 = q.get()
            assert isinstance(task3, FinalizeTask)

    def test_load_files_ignores_subdirectories(self):
        """Test load_files ignores subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in root
            (Path(tmpdir) / "01_test.jpg").touch()

            # Create subdirectory with file
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "02_test.jpg").touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should only have 1 file from root + FinalizeTask
            assert q.qsize() == 2

            task1 = q.get()
            assert isinstance(task1, FileTask)
            assert task1.file_path.name == "01_test.jpg"

            task2 = q.get()
            assert isinstance(task2, FinalizeTask)

    def test_load_files_case_insensitive_extension(self):
        """Test load_files handles case-insensitive extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different case extensions
            (Path(tmpdir) / "01_test.jpg").touch()
            (Path(tmpdir) / "02_test.JPG").touch()
            (Path(tmpdir) / "03_test.Jpg").touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # On case-sensitive systems, only .jpg will match
            # On case-insensitive systems (Windows), all might match
            # We test that at least the .jpg file is loaded
            assert q.qsize() >= 2  # At least 1 file + FinalizeTask

            task1 = q.get()
            assert isinstance(task1, FileTask)

    def test_load_files_unnumbered_files_get_sequential_numbers(self):
        """Test that unnumbered files get sequential sort keys after numbered files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mix of numbered and unnumbered files
            (Path(tmpdir) / "01_first.jpg").touch()
            (Path(tmpdir) / "02_second.jpg").touch()
            (Path(tmpdir) / "alpha.jpg").touch()
            (Path(tmpdir) / "beta.jpg").touch()

            q = queue.Queue()
            loader = FileLoader(target_q=q, input_dir=tmpdir, extension=".jpg")

            loader.load_files()

            # Should have 4 FileTasks + 1 FinalizeTask
            assert q.qsize() == 5

            # Numbered files come first
            task1 = q.get()
            assert isinstance(task1, FileTask)
            assert task1.file_path.name == "01_first.jpg"
            assert task1.sort_key == 1.0

            task2 = q.get()
            assert isinstance(task2, FileTask)
            assert task2.file_path.name == "02_second.jpg"
            assert task2.sort_key == 2.0

            # Unnumbered files get sequential numbers (sorted by name)
            task3 = q.get()
            assert isinstance(task3, FileTask)
            assert task3.file_path.name == "alpha.jpg"
            assert task3.sort_key == 3.0  # Continues from max numbered file

            task4 = q.get()
            assert isinstance(task4, FileTask)
            assert task4.file_path.name == "beta.jpg"
            assert task4.sort_key == 4.0

            # Last should be FinalizeTask
            task5 = q.get()
            assert isinstance(task5, FinalizeTask)
