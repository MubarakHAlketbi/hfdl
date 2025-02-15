import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from hfdl.speed_manager import (
    SpeedManager,
    SpeedMeasurement,
    SpeedManagerError,
    SpeedMeasurementError,
    SpeedAllocationError
)

@pytest.fixture
def mock_api():
    """Create a mock HfApi instance"""
    api = Mock(spec=HfApi)
    return api

@pytest.fixture
def speed_manager(mock_api):
    """Create a SpeedManager instance with mock API"""
    return SpeedManager(
        api=mock_api,
        measure_duration=2,
        bandwidth_percentage=95.0,
        chunk_size=8192
    )

def test_initialization_errors():
    """Test error handling during initialization"""
    api = Mock(spec=HfApi)
    
    # Test invalid measure duration
    with pytest.raises(SpeedManagerError, match="Invalid parameter"):
        SpeedManager(api=api, measure_duration=0, bandwidth_percentage=95, chunk_size=8192)
    
    # Test invalid bandwidth percentage
    with pytest.raises(SpeedManagerError, match="Invalid parameter"):
        SpeedManager(api=api, measure_duration=2, bandwidth_percentage=101, chunk_size=8192)
    
    # Test invalid chunk size
    with pytest.raises(SpeedManagerError, match="Invalid parameter"):
        SpeedManager(api=api, measure_duration=2, bandwidth_percentage=95, chunk_size=0)

def test_speed_measurement_calculation():
    """Test speed measurement calculations"""
    # Test valid measurement
    measurement = SpeedMeasurement(
        timestamp=time.time(),
        bytes_transferred=1024 * 1024,  # 1MB
        duration=2.0  # 2 seconds
    )
    assert measurement.bytes_per_second == 524288  # 1MB / 2s = 524,288 B/s
    
    # Test zero duration handling
    measurement = SpeedMeasurement(
        timestamp=time.time(),
        bytes_transferred=1024,
        duration=0
    )
    assert measurement.bytes_per_second == 0

def test_repository_errors(mock_api, speed_manager):
    """Test handling of repository-related errors"""
    # Test repository not found
    mock_api.hf_hub_url.side_effect = RepositoryNotFoundError("Repo not found")
    with pytest.raises(RepositoryNotFoundError):
        speed_manager.measure_initial_speed("test/repo", "test.bin")
    
    # Test file not found
    mock_api.hf_hub_url.side_effect = EntryNotFoundError("File not found")
    with pytest.raises(EntryNotFoundError):
        speed_manager.measure_initial_speed("test/repo", "test.bin")

def test_network_errors(mock_api, speed_manager):
    """Test handling of network errors"""
    mock_api.hf_hub_url.return_value = "https://test.com/file"
    
    # Test HTTP error
    with patch('requests.get', side_effect=HTTPError("HTTP error")):
        with pytest.raises(HTTPError):
            speed_manager.measure_initial_speed("test/repo", "test.bin")
    
    # Test connection error
    with patch('requests.get', side_effect=ConnectionError("Connection failed")):
        with pytest.raises(ConnectionError):
            speed_manager.measure_initial_speed("test/repo", "test.bin")
    
    # Test timeout
    with patch('requests.get', side_effect=Timeout("Request timed out")):
        with pytest.raises(Timeout):
            speed_manager.measure_initial_speed("test/repo", "test.bin")

def test_measurement_errors(mock_api, speed_manager):
    """Test handling of speed measurement errors"""
    mock_api.hf_hub_url.return_value = "https://test.com/file"
    
    # Mock response with no content
    mock_response = MagicMock()
    mock_response.iter_content.return_value = []
    
    with patch('requests.get', return_value=mock_response):
        with pytest.raises(SpeedMeasurementError, match="Speed calculation failed"):
            speed_manager.measure_initial_speed("test/repo", "test.bin")

def test_allocation_errors(speed_manager):
    """Test handling of speed allocation errors"""
    # Test allocation without measurement
    with pytest.raises(SpeedAllocationError, match="Invalid parameters"):
        speed_manager.allocate_thread_speed(
            thread_id=1,
            file_size=1024,
            total_size=1024,
            remaining_files=1,
            total_files=1
        )
    
    # Set allowed speed for testing
    speed_manager._allowed_speed = 1000000  # 1MB/s
    
    # Test negative file size
    with pytest.raises(SpeedAllocationError, match="Invalid parameters"):
        speed_manager.allocate_thread_speed(
            thread_id=1,
            file_size=-1,
            total_size=1024,
            remaining_files=1,
            total_files=1
        )
    
    # Test negative total size
    with pytest.raises(SpeedAllocationError, match="Invalid parameters"):
        speed_manager.allocate_thread_speed(
            thread_id=1,
            file_size=1024,
            total_size=-1,
            remaining_files=1,
            total_files=1
        )
    
    # Test invalid file counts
    with pytest.raises(SpeedAllocationError, match="Invalid parameters"):
        speed_manager.allocate_thread_speed(
            thread_id=1,
            file_size=1024,
            total_size=1024,
            remaining_files=2,  # More remaining than total
            total_files=1
        )

def test_successful_speed_measurement(mock_api, speed_manager):
    """Test successful speed measurement"""
    mock_api.hf_hub_url.return_value = "https://test.com/file"
    
    # Mock successful download
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [
        b"x" * 8192 for _ in range(10)  # 10 chunks of 8KB
    ]
    
    with patch('requests.get', return_value=mock_response):
        speed = speed_manager.measure_initial_speed(
            repo_id="test/repo",
            sample_file="test.bin"
        )
        
        # Verify speed calculation
        assert speed > 0
        assert speed == speed_manager.allowed_speed
        assert speed == speed_manager._allowed_speed * 0.95  # 95% of measured speed

def test_thread_speed_allocation(speed_manager):
    """Test thread speed allocation"""
    # Set allowed speed
    speed_manager._allowed_speed = 1000000  # 1MB/s
    
    # Allocate speed for multiple threads
    speeds = []
    for i in range(3):
        speed = speed_manager.allocate_thread_speed(
            thread_id=i,
            file_size=1024 * 1024,  # 1MB
            total_size=3 * 1024 * 1024,  # 3MB
            remaining_files=3-i,
            total_files=3
        )
        speeds.append(speed)
    
    # Verify allocations
    assert all(s > 0 for s in speeds)
    assert sum(speeds) > 0

def test_thread_safety(speed_manager):
    """Test thread-safe operations"""
    import threading
    
    # Set allowed speed
    speed_manager._allowed_speed = 1000000  # 1MB/s
    
    def allocate_speed():
        """Allocate speed in thread"""
        for i in range(10):
            try:
                speed_manager.allocate_thread_speed(
                    thread_id=i,
                    file_size=1024,
                    total_size=10240,
                    remaining_files=10-i,
                    total_files=10
                )
            except SpeedAllocationError:
                continue
    
    # Create threads
    threads = [
        threading.Thread(target=allocate_speed)
        for _ in range(5)
    ]
    
    # Run threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify thread speeds are recorded
    assert len(speed_manager._thread_speeds) > 0