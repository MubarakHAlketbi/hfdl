import multiprocessing
import signal
import threading
from enum import Enum
from typing import Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ThreadManagerError(Exception):
    """Base exception for thread manager errors"""
    pass

class ThreadScenario(Enum):
    """Enumeration of possible thread allocation scenarios"""
    SINGLE_THREAD = 1    # All resources for download
    DUAL_THREAD = 2      # One for ctrl+c, one for download
    TRIPLE_THREAD = 3    # One free, one for ctrl+c, one for download
    MULTI_THREAD = 4     # One free, one for ctrl+c, rest for download

class ThreadManager:
    """Manages thread allocation and control based on CPU availability"""
    
    def __init__(self):
        try:
            self._cpu_count = multiprocessing.cpu_count()
            self._scenario = self._determine_scenario()
            self._stop_event = threading.Event()
            self._executor: Optional[ThreadPoolExecutor] = None
            self._ctrl_c_handler: Optional[threading.Thread] = None
        except Exception as e:
            logger.error(f"Failed to initialize thread manager: {e}")
            raise ThreadManagerError(f"Thread manager initialization failed: {e}")
            
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
        try:
            def signal_handler(signum, frame):
                logger.info("Received stop signal, initiating graceful shutdown...")
                self._stop_event.set()
                
            def ctrl_c_thread():
                try:
                    signal.signal(signal.SIGINT, signal_handler)
                    # Keep thread alive
                    while not self._stop_event.is_set():
                        self._stop_event.wait(1)
                except Exception as e:
                    logger.error(f"Error in ctrl+c handler thread: {e}")
                    self._stop_event.set()
                    
            # Only setup handler if we have more than one thread
            if self._scenario != ThreadScenario.SINGLE_THREAD:
                try:
                    self._ctrl_c_handler = threading.Thread(
                        target=ctrl_c_thread,
                        name="ctrl_c_handler",
                        daemon=True
                    )
                    self._ctrl_c_handler.start()
                except threading.ThreadError as e:
                    logger.error(f"Failed to start ctrl+c handler thread: {e}")
                    raise ThreadManagerError(f"Failed to start ctrl+c handler: {e}")
        except Exception as e:
            logger.error(f"Failed to setup ctrl+c handler: {e}")
            raise ThreadManagerError(f"Failed to setup ctrl+c handler: {e}")
            
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
        try:
            self._setup_ctrl_c_handler()
            max_workers = self.get_download_threads()
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="download_worker"
            )
            logger.info(f"Thread manager started with {max_workers} download worker(s)")
        except Exception as e:
            logger.error(f"Failed to start thread pool: {e}")
            raise ThreadManagerError(f"Failed to start thread pool: {e}")
        except Exception as e:
            logger.error(f"Failed to start thread manager: {e}")
            raise ThreadManagerError(f"Failed to start thread manager: {e}")
        
    def stop(self):
        """Gracefully shutdown thread manager and all components"""
        try:
            if self._executor:
                self._stop_event.set()
                self._executor.shutdown(wait=True)
                self._executor = None
            logger.info("Thread manager stopped")
        except Exception as e:
            logger.error(f"Error during thread manager shutdown: {e}")
            # Still set executor to None to prevent further usage
            self._executor = None
            raise ThreadManagerError(f"Error during shutdown: {e}")
        
    def submit_download(self, fn: Callable, *args, **kwargs):
        """Submit download task to thread pool
        
        Args:
            fn: Download function to execute
            *args: Positional arguments for download function
            **kwargs: Keyword arguments for download function
        
        Returns:
            Future object representing the download task
        
        Raises:
            ThreadManagerError: If thread manager not started or submission fails
        """
        if not self._executor:
            raise ThreadManagerError("Thread manager not started")
            
        try:
            return self._executor.submit(fn, *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to submit download task: {e}")
            raise ThreadManagerError(f"Failed to submit download task: {e}")
        
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