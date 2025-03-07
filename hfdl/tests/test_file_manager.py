import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from requests.exceptions import HTTPError
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    EntryNotFoundError,
    RevisionNotFoundError
)
from huggingface_hub.hf_api import RepoFile
from hfdl.file_manager import (
    FileManager,
    FileInfo,
    FileManagerError,
    FileSizeError,
    FileTrackingError
)

@pytest.fixture
def mock_api():
    """Create a mock HfApi instance"""
    api = Mock(spec=HfApi)
    return api

@pytest.fixture
def file_manager(mock_api):
    """Create a FileManager instance with mock API"""
    return FileManager(api=mock_api, size_threshold_mb=100.0)

def test_initialization_error():
    """Test error handling during initialization"""
    # Test negative threshold
    with pytest.raises(FileManagerError, match="initialization failed"):
        FileManager(api=Mock(spec=HfApi), size_threshold_mb=-1)

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

def test_repository_not_found(mock_api, file_manager):
    """Test handling of repository not found error"""
    mock_api.list_repo_tree.side_effect = RepositoryNotFoundError("Not found")
    
    with pytest.raises(RepositoryNotFoundError):
        file_manager.discover_files(
            repo_id="nonexistent/repo",
            repo_type="model"
        )

def test_revision_not_found(mock_api, file_manager):
    """Test handling of revision not found error"""
    mock_api.list_repo_tree.side_effect = RevisionNotFoundError("Bad revision")
    
    with pytest.raises(RevisionNotFoundError):
        file_manager.discover_files(
            repo_id="test/repo",
            repo_type="model"
        )

def test_network_error(mock_api, file_manager):
    """Test handling of network errors"""
    mock_api.list_repo_tree.side_effect = HTTPError("Network failed")
    
    with pytest.raises(HTTPError):
        file_manager.discover_files(
            repo_id="test/repo",
            repo_type="model"
        )

def test_metadata_error(mock_api, file_manager):
    """Test handling of metadata retrieval errors"""
    # This test is no longer needed since we're using list_repo_tree
    # which provides metadata directly, but we'll keep it for completeness
    # by testing a different error case
    
    # Mock repo_tree with a malformed RepoFile (missing required attributes)
    class MalformedRepoFile:
        def __init__(self):
            self.path = "malformed.txt"
            # Missing size and type attributes
    
    mock_api.list_repo_tree.return_value = [MalformedRepoFile()]
    
    # Should raise an error when trying to access missing attributes
    with pytest.raises(FileSizeError, match="Failed to process file info"):
        file_manager.discover_files(
            repo_id="test/repo",
            repo_type="model"
        )

def test_file_size_error(mock_api, file_manager):
    """Test handling of file size calculation errors"""
    # Mock repo_tree with a RepoFile that has invalid size
    mock_api.list_repo_tree.return_value = [
        RepoFile(path="invalid.txt", size=-1, type="file")  # Invalid negative size
    ]
    
    with pytest.raises(FileSizeError, match="Failed to process file info"):
        file_manager.discover_files(
            repo_id="test/repo",
            repo_type="model"
        )

def test_progress_tracking_error(file_manager):
    """Test error handling in progress tracking"""
    # Test negative progress
    with pytest.raises(FileTrackingError, match="Invalid progress value"):
        file_manager.update_progress("test.txt", -1)

def test_progress_calculation_error(file_manager):
    """Test error handling in progress calculations"""
    # Create a file that will cause calculation error
    file_info = FileInfo(
        name="test.txt",
        size=1024,
        path_in_repo="test.txt",
        local_path=Path("test.txt")
    )
    
    with file_manager._lock:
        file_manager._files["test.txt"] = file_info
    
    # Test with valid progress
    file_manager.update_progress("test.txt", 512)
    progress, total = file_manager.get_progress("test.txt")
    assert progress == 512
    assert total == 1024
    
    # Test completion
    file_manager.update_progress("test.txt", 1024)
    assert file_manager.is_completed("test.txt")

def test_total_progress_error(file_manager):
    """Test error handling in total progress calculation"""
    # Create files with valid and invalid sizes
    files = {
        "valid.txt": FileInfo(
            name="valid.txt",
            size=1000,
            path_in_repo="valid.txt",
            local_path=Path("valid.txt"),
            downloaded=500
        ),
        "invalid.txt": FileInfo(
            name="invalid.txt",
            size=-1,  # Invalid size
            path_in_repo="invalid.txt",
            local_path=Path("invalid.txt")
        )
    }
    
    with file_manager._lock:
        file_manager._files = files
    
    # Should handle invalid sizes gracefully
    total_downloaded, total_size = file_manager.get_total_progress()
    assert total_downloaded == 500  # Only from valid file
    assert total_size == 1000  # Only from valid file

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
            try:
                file_manager.update_progress("test.txt", progress)
            except FileTrackingError:
                continue  # Ignore invalid progress values
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

def test_file_categorization(mock_api, file_manager):
    """Test file categorization based on size"""
    # Mock repo_tree with RepoFile objects
    mock_api.list_repo_tree.return_value = [
        RepoFile(path="small.txt", size=50 * 1024 * 1024, type="file"),  # 50MB
        RepoFile(path="big.txt", size=200 * 1024 * 1024, type="file"),   # 200MB
        RepoFile(path="folder", size=0, type="directory")  # Directory should be skipped
    ]
    
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
    
    # Verify directory was skipped
    assert all(f.name != "folder" for f in small_files + big_files)