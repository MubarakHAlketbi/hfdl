import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from huggingface_hub import HfApi
import requests
from hfdl.speed_manager import SpeedManager, SpeedMeasurement

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

def test_speed_measurement_calculation():
    """Test speed measurement calculations"""
    measurement = SpeedMeasurement(
        timestamp=time.time(),
        bytes_transferred=1024 * 1024,  # 1MB
        duration=2.0  # 2 seconds
    )
    
    # Should be 524,288 bytes per second (1MB / 2s)
    assert measurement.bytes_per_second == 524288

def test_speed_measurement_zero_duration():
    """Test speed measurement with zero duration"""
    measurement = SpeedMeasurement(
        timestamp=time.time(),
        bytes_transferred=1024,
        duration=0
    )
    
    # Should handle zero duration gracefully
    assert measurement.bytes_per_second == 0

def test_initial_speed_measurement(mock_api, speed_manager):
    """Test initial speed measurement"""
    # Mock API URL generation
    mock_api.hf_hub_url.return_value = "https://test.com/file"
    
    # Mock requests.get
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [
        b"x" * 8192 for _ in range(10)  # 10 chunks of 8KB
    ]
    
    with patch("requests.get", return_value=mock_response):
        speed = speed_manager.measure_initial_speed(
            repo_id="test/repo",
            sample_file="test.bin"
        )
        
        # Verify speed calculation
        assert speed > 0
        # Should be 95% of measured speed
        assert speed == speed_manager.allowed_speed

def test_thread_speed_allocation(speed_manager):
    """Test thread speed allocation"""
    # Set allowed speed
    speed_manager._allowed_speed = 1000000  # 1MB/s
    
    # Allocate speed for a thread
    thread_speed = speed_manager.allocate_thread_speed(
        thread_id=1,
        file_size=1024 * 1024,  # 1MB
        total_size=2 * 1024 * 1024,  # 2MB
        remaining_files=2,
        total_files=3
    )
    
    # Verify allocation
    assert thread_speed > 0
    assert thread_speed <= speed_manager.allowed_speed

def test_multiple_thread_allocation(speed_manager):
    """Test speed allocation across multiple threads"""
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

def test_get_thread_speed(speed_manager):
    """Test getting allocated thread speed"""
    # Set allowed speed
    speed_manager._allowed_speed = 1000000  # 1MB/s
    
    # Allocate speed
    allocated_speed = speed_manager.allocate_thread_speed(
        thread_id=1,
        file_size=1024,
        total_size=1024,
        remaining_files=1,
        total_files=1
    )
    
    # Get allocated speed
    thread_speed = speed_manager.get_thread_speed(1)
    assert thread_speed == allocated_speed

def test_get_nonexistent_thread_speed(speed_manager):
    """Test getting speed for non-existent thread"""
    assert speed_manager.get_thread_speed(999) is None

def test_speed_measurement_error_handling(mock_api, speed_manager):
    """Test error handling in speed measurement"""
    # Mock API error
    mock_api.hf_hub_url.side_effect = Exception("API Error")
    
    with pytest.raises(ValueError, match="Speed measurement failed"):
        speed_manager.measure_initial_speed(
            repo_id="test/repo",
            sample_file="test.bin"
        )

def test_allocation_without_measurement(speed_manager):
    """Test allocation without speed measurement"""
    with pytest.raises(ValueError, match="No speed measurement available"):
        speed_manager.allocate_thread_speed(
            thread_id=1,
            file_size=1024,
            total_size=1024,
            remaining_files=1,
            total_files=1
        )

def test_thread_safety(speed_manager):
    """Test thread-safe operations"""
    import threading
    
    # Set allowed speed
    speed_manager._allowed_speed = 1000000  # 1MB/s
    
    def allocate_speed():
        """Allocate speed in thread"""
        for i in range(10):
            speed_manager.allocate_thread_speed(
                thread_id=i,
                file_size=1024,
                total_size=10240,
                remaining_files=10-i,
                total_files=10
            )
    
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