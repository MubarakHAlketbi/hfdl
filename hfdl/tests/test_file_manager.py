import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from hfdl.file_manager import FileManager, FileInfo

@pytest.fixture
def mock_api():
    """Create a mock HfApi instance"""
    api = Mock(spec=HfApi)
    return api

@pytest.fixture
def file_manager(mock_api):
    """Create a FileManager instance with mock API"""
    return FileManager(api=mock_api, size_threshold_mb=100.0)

def test_file_info_creation():
    """Test FileInfo dataclass creation"""
    file_info = FileInfo(
        name="test.txt",
        size=1024,
        path_in_repo="path/test.txt",
        local_path=Path("test.txt")
    )
    
    assert file_info.name == "test.txt"
    assert file_info.size == 1024
    assert file_info.path_in_repo == "path/test.txt"
    assert file_info.local_path == Path("test.txt")
    assert file_info.downloaded == 0
    assert file_info.completed is False

def test_file_categorization(mock_api, file_manager):
    """Test file categorization based on size"""
    # Mock file list
    mock_api.list_repo_files.return_value = [
        "small.txt",
        "big.txt"
    ]
    
    # Mock file metadata
    def mock_metadata(repo_id, repo_type, path, token):
        if path == "small.txt":
            return Mock(size=50 * 1024 * 1024)  # 50MB
        else:
            return Mock(size=200 * 1024 * 1024)  # 200MB
    
    mock_api.get_repo_file_metadata.side_effect = mock_metadata
    
    # Get categorized files
    small_files, big_files = file_manager.discover_files(
        repo_id="test/repo",
        repo_type="model"
    )
    
    # Verify categorization
    assert len(small_files) == 1
    assert len(big_files) == 1
    assert small_files[0].name == "small.txt"
    assert big_files[0].name == "big.txt"

def test_progress_tracking(file_manager):
    """Test download progress tracking"""
    # Create test file
    file_info = FileInfo(
        name="test.txt",
        size=1000,
        path_in_repo="test.txt",
        local_path=Path("test.txt")
    )
    
    # Add file to manager
    with file_manager._lock:
        file_manager._files["test.txt"] = file_info
    
    # Update progress
    file_manager.update_progress("test.txt", 500)
    
    # Check progress
    downloaded, total = file_manager.get_progress("test.txt")
    assert downloaded == 500
    assert total == 1000
    assert not file_manager.is_completed("test.txt")
    
    # Complete download
    file_manager.update_progress("test.txt", 1000)
    assert file_manager.is_completed("test.txt")

def test_total_progress(file_manager):
    """Test total progress calculation"""
    # Create test files
    files = {
        "file1.txt": FileInfo(
            name="file1.txt",
            size=1000,
            path_in_repo="file1.txt",
            local_path=Path("file1.txt"),
            downloaded=500
        ),
        "file2.txt": FileInfo(
            name="file2.txt",
            size=2000,
            path_in_repo="file2.txt",
            local_path=Path("file2.txt"),
            downloaded=1000
        )
    }
    
    # Add files to manager
    with file_manager._lock:
        file_manager._files = files
    
    # Check total progress
    total_downloaded, total_size = file_manager.get_total_progress()
    assert total_downloaded == 1500  # 500 + 1000
    assert total_size == 3000  # 1000 + 2000

def test_repository_not_found(mock_api, file_manager):
    """Test handling of repository not found error"""
    mock_api.list_repo_files.side_effect = RepositoryNotFoundError("Not found")
    
    with pytest.raises(RepositoryNotFoundError):
        file_manager.discover_files(
            repo_id="nonexistent/repo",
            repo_type="model"
        )

def test_thread_safety(file_manager):
    """Test thread-safe operations"""
    import threading
    import random
    import time
    
    # Create test file
    file_info = FileInfo(
        name="test.txt",
        size=1000,
        path_in_repo="test.txt",
        local_path=Path("test.txt")
    )
    
    with file_manager._lock:
        file_manager._files["test.txt"] = file_info
    
    def update_progress():
        """Randomly update progress"""
        for _ in range(10):
            progress = random.randint(0, 1000)
            file_manager.update_progress("test.txt", progress)
            time.sleep(0.01)
    
    # Create threads
    threads = [
        threading.Thread(target=update_progress)
        for _ in range(5)
    ]
    
    # Run threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify final state is consistent
    downloaded, total = file_manager.get_progress("test.txt")
    assert 0 <= downloaded <= total