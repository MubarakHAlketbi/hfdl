import pytest
import threading
import signal
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, Mock
from hfdl.thread_manager import ThreadManager, ThreadManagerError, ThreadScenario

def test_thread_scenario_detection():
    """Test thread scenario detection based on CPU count"""
    manager = ThreadManager()
    cpu_count = manager._cpu_count
    
    # Verify scenario matches CPU count
    if cpu_count == 1:
        assert manager._scenario == ThreadScenario.SINGLE_THREAD
    elif cpu_count == 2:
        assert manager._scenario == ThreadScenario.DUAL_THREAD
    elif cpu_count == 3:
        assert manager._scenario == ThreadScenario.TRIPLE_THREAD
    else:
        assert manager._scenario == ThreadScenario.MULTI_THREAD

def test_initialization_error():
    """Test error handling during initialization"""
    with patch('multiprocessing.cpu_count', side_effect=OSError("CPU count error")):
        with pytest.raises(ThreadManagerError, match="Thread manager initialization failed"):
            ThreadManager()

def test_signal_handler_error():
    """Test error handling in signal handler setup"""
    manager = ThreadManager()
    
    # Mock signal.signal to raise error
    with patch('signal.signal', side_effect=ValueError("Signal error")):
        with pytest.raises(ThreadManagerError, match="Failed to setup ctrl+c handler"):
            manager._setup_ctrl_c_handler()

def test_thread_creation_error():
    """Test error handling in thread creation"""
    manager = ThreadManager()
    
    # Mock threading.Thread to raise error
    with patch('threading.Thread', side_effect=RuntimeError("Thread creation failed")):
        with pytest.raises(ThreadManagerError, match="Failed to start ctrl+c handler"):
            manager._setup_ctrl_c_handler()

def test_thread_pool_error():
    """Test error handling in thread pool creation"""
    manager = ThreadManager()
    
    # Mock ThreadPoolExecutor to raise error
    with patch('concurrent.futures.ThreadPoolExecutor', side_effect=RuntimeError("Pool creation failed")):
        with pytest.raises(ThreadManagerError, match="Failed to start thread pool"):
            manager.start()

def test_submit_without_start():
    """Test submitting download without starting manager"""
    manager = ThreadManager()
    
    with pytest.raises(ThreadManagerError, match="Thread manager not started"):
        manager.submit_download(lambda: None)

def test_submit_error():
    """Test error handling in task submission"""
    with ThreadManager() as manager:
        # Mock submit to raise error
        with patch.object(manager._executor, 'submit', side_effect=RuntimeError("Submit failed")):
            with pytest.raises(ThreadManagerError, match="Failed to submit download task"):
                manager.submit_download(lambda: None)

def test_shutdown_error():
    """Test error handling during shutdown"""
    manager = ThreadManager()
    manager.start()
    
    # Mock shutdown to raise error
    with patch.object(manager._executor, 'shutdown', side_effect=RuntimeError("Shutdown failed")):
        with pytest.raises(ThreadManagerError, match="Error during shutdown"):
            manager.stop()
        # Verify executor is still set to None
        assert manager._executor is None

def test_ctrl_c_handler_error():
    """Test error handling in ctrl+c handler thread"""
    manager = ThreadManager()
    
    # Create a mock thread that raises an error
    mock_thread = Mock()
    mock_thread.start.side_effect = RuntimeError("Handler thread failed")
    
    with patch('threading.Thread', return_value=mock_thread):
        with pytest.raises(ThreadManagerError, match="Failed to start ctrl+c handler"):
            manager._setup_ctrl_c_handler()

def test_download_threads_calculation():
    """Test number of download threads calculation"""
    manager = ThreadManager()
    threads = manager.get_download_threads()
    
    # Verify thread count based on scenario
    if manager._scenario == ThreadScenario.SINGLE_THREAD:
        assert threads == 1
    elif manager._scenario == ThreadScenario.DUAL_THREAD:
        assert threads == 1
    elif manager._scenario == ThreadScenario.TRIPLE_THREAD:
        assert threads == 1
    else:
        # For multi-thread, should be cpu_count - 2
        assert threads == max(1, manager._cpu_count - 2)

def test_context_manager():
    """Test thread manager context handling"""
    with ThreadManager() as manager:
        # Should be started
        assert not manager.should_stop
        assert manager._executor is not None
        
        # Should have correct number of workers
        max_workers = manager._executor._max_workers
        assert max_workers == manager.get_download_threads()
    
    # After context exit
    assert manager.should_stop
    assert manager._executor is None

def test_graceful_shutdown():
    """Test graceful shutdown with running tasks"""
    def long_task():
        """Simulate a long-running task"""
        import time
        time.sleep(0.1)
        return "completed"
    
    with ThreadManager() as manager:
        # Submit a task
        future = manager.submit_download(long_task)
        
        # Verify task completes despite shutdown
        assert future.result() == "completed"
    
    # Verify clean shutdown
    assert manager._executor is None
    assert manager.should_stop

def test_multiple_stop_calls():
    """Test multiple stop calls are handled safely"""
    manager = ThreadManager()
    manager.start()
    
    # Multiple stops should not raise errors
    manager.stop()
    manager.stop()
    manager.stop()
    
    assert manager._executor is None
    assert manager.should_stop