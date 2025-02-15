import pytest
import multiprocessing
from hfdl.thread_manager import ThreadManager, ThreadScenario

def test_thread_scenario_detection():
    """Test thread scenario detection based on CPU count"""
    manager = ThreadManager()
    cpu_count = multiprocessing.cpu_count()
    
    # Get actual scenario
    scenario = manager._determine_scenario()
    
    # Verify scenario matches CPU count
    if cpu_count == 1:
        assert scenario == ThreadScenario.SINGLE_THREAD
    elif cpu_count == 2:
        assert scenario == ThreadScenario.DUAL_THREAD
    elif cpu_count == 3:
        assert scenario == ThreadScenario.TRIPLE_THREAD
    else:
        assert scenario == ThreadScenario.MULTI_THREAD

def test_download_threads_calculation():
    """Test number of download threads calculation"""
    manager = ThreadManager()
    cpu_count = multiprocessing.cpu_count()
    threads = manager.get_download_threads()
    
    # Verify thread count based on scenario
    if cpu_count <= 3:
        assert threads == 1
    else:
        # For multi-thread, should be cpu_count - 2
        assert threads == max(1, cpu_count - 2)

def test_thread_manager_context():
    """Test thread manager context handling"""
    with ThreadManager() as manager:
        # Should be started
        assert not manager.should_stop
        
        # Should have executor
        assert manager._executor is not None
        
        # Should have correct number of workers
        max_workers = manager._executor._max_workers
        assert max_workers == manager.get_download_threads()
    
    # After context exit
    assert manager.should_stop
    assert manager._executor is None

def test_ctrl_c_handler():
    """Test ctrl+c handler setup"""
    manager = ThreadManager()
    
    # Start manager
    manager.start()
    
    try:
        # Check handler thread based on scenario
        if manager._scenario == ThreadScenario.SINGLE_THREAD:
            assert manager._ctrl_c_handler is None
        else:
            assert manager._ctrl_c_handler is not None
            assert manager._ctrl_c_handler.is_alive()
    finally:
        # Clean up
        manager.stop()

def test_submit_download():
    """Test download task submission"""
    def dummy_download():
        return "downloaded"
    
    with ThreadManager() as manager:
        # Submit task
        future = manager.submit_download(dummy_download)
        
        # Wait for result
        result = future.result()
        assert result == "downloaded"

def test_submit_without_start():
    """Test submitting download without starting manager"""
    manager = ThreadManager()
    
    # Should raise error if not started
    with pytest.raises(RuntimeError, match="Thread manager not started"):
        manager.submit_download(lambda: None)