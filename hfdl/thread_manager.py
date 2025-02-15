import multiprocessing
import signal
import threading
from enum import Enum
from typing import Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ThreadScenario(Enum):
    """Enumeration of possible thread allocation scenarios"""
    SINGLE_THREAD = 1    # All resources for download
    DUAL_THREAD = 2      # One for ctrl+c, one for download
    TRIPLE_THREAD = 3    # One free, one for ctrl+c, one for download
    MULTI_THREAD = 4     # One free, one for ctrl+c, rest for download

class ThreadManager:
    """Manages thread allocation and control based on CPU availability
    
    Handles:
    - CPU thread detection
    - Scenario selection
    - Signal handling
    - Thread pool management
    """
    
    def __init__(self):
        self._cpu_count = multiprocessing.cpu_count()
        self._scenario = self._determine_scenario()
        self._stop_event = threading.Event()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._ctrl_c_handler: Optional[threading.Thread] = None
        
    def _determine_scenario(self) -> ThreadScenario:
        """Determine thread allocation scenario based on CPU count"""
        if self._cpu_count == 1:
            return ThreadScenario.SINGLE_THREAD
        elif self._cpu_count == 2:
            return ThreadScenario.DUAL_THREAD
        elif self._cpu_count == 3:
            return ThreadScenario.TRIPLE_THREAD
        else:
            return ThreadScenario.MULTI_THREAD
            
    def _setup_ctrl_c_handler(self):
        """Setup dedicated thread for handling ctrl+c"""
        def signal_handler(signum, frame):
            logger.info("Received stop signal, initiating graceful shutdown...")
            self._stop_event.set()
            
        def ctrl_c_thread():
            signal.signal(signal.SIGINT, signal_handler)
            # Keep thread alive
            while not self._stop_event.is_set():
                self._stop_event.wait(1)
                
        # Only setup handler if we have more than one thread
        if self._scenario != ThreadScenario.SINGLE_THREAD:
            self._ctrl_c_handler = threading.Thread(
                target=ctrl_c_thread,
                name="ctrl_c_handler",
                daemon=True
            )
            self._ctrl_c_handler.start()
            
    def get_download_threads(self) -> int:
        """Get number of threads available for download operations"""
        if self._scenario == ThreadScenario.SINGLE_THREAD:
            return 1
        elif self._scenario == ThreadScenario.DUAL_THREAD:
            return 1
        elif self._scenario == ThreadScenario.TRIPLE_THREAD:
            return 1
        else:
            # For multi-thread, reserve 2 threads (free + ctrl_c)
            return max(1, self._cpu_count - 2)
            
    def start(self):
        """Initialize thread manager and start necessary components"""
        self._setup_ctrl_c_handler()
        max_workers = self.get_download_threads()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="download_worker"
        )
        logger.info(f"Thread manager started with {max_workers} download worker(s)")
        
    def stop(self):
        """Gracefully shutdown thread manager and all components"""
        if self._executor:
            self._stop_event.set()
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("Thread manager stopped")
        
    def submit_download(self, fn: Callable, *args, **kwargs):
        """Submit download task to thread pool
        
        Args:
            fn: Download function to execute
            *args: Positional arguments for download function
            **kwargs: Keyword arguments for download function
        
        Returns:
            Future object representing the download task
        
        Raises:
            RuntimeError: If thread manager not started
        """
        if not self._executor:
            raise RuntimeError("Thread manager not started")
        return self._executor.submit(fn, *args, **kwargs)
        
    @property
    def should_stop(self) -> bool:
        """Check if stop signal has been received"""
        return self._stop_event.is_set()
        
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()