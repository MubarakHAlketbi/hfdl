import time
import threading
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from statistics import mean
import requests
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

logger = logging.getLogger(__name__)

class SpeedManagerError(Exception):
    """Base exception for speed manager errors"""
    pass

class SpeedMeasurementError(SpeedManagerError):
    """Error during speed measurement"""
    pass

class SpeedAllocationError(SpeedManagerError):
    """Error during speed allocation"""
    pass

@dataclass
class SpeedMeasurement:
    """Represents a speed measurement sample"""
    timestamp: float
    bytes_transferred: int
    duration: float
    
    @property
    def bytes_per_second(self) -> float:
        """Calculate bytes per second for this measurement"""
        try:
            if self.duration <= 0:
                raise ValueError("Duration must be positive")
            return self.bytes_transferred / self.duration
        except Exception as e:
            logger.error(f"Error calculating speed: {e}")
            return 0

class SpeedManager:
    """Manages download speed measurement and control"""
    
    def __init__(
        self,
        api: HfApi,
        measure_duration: int,
        bandwidth_percentage: float,
        chunk_size: int
    ):
        try:
            if measure_duration <= 0:
                raise ValueError("Measure duration must be positive")
            if not 0 < bandwidth_percentage <= 100:
                raise ValueError("Bandwidth percentage must be between 0 and 100")
            if chunk_size <= 0:
                raise ValueError("Chunk size must be positive")
                
            self.api = api
            self.measure_duration = measure_duration
            self.bandwidth_percentage = bandwidth_percentage / 100.0
            self.chunk_size = chunk_size
            
            self._measurements: List[SpeedMeasurement] = []
            self._thread_speeds: Dict[int, float] = {}
            self._lock = threading.Lock()
            self._allowed_speed: Optional[float] = None
            
        except ValueError as e:
            logger.error(f"Invalid initialization parameter: {e}")
            raise SpeedManagerError(f"Invalid parameter: {e}")
        except Exception as e:
            logger.error(f"Error initializing speed manager: {e}")
            raise SpeedManagerError(f"Initialization failed: {e}")
        
    def measure_initial_speed(
        self,
        repo_id: str,
        sample_file: str,
        token: Optional[str] = None
    ) -> float:
        """Measure initial download speed using a sample file
        
        Args:
            repo_id: Repository identifier
            sample_file: Path to sample file in repository
            token: Optional authentication token
            
        Returns:
            Measured speed in bytes per second
            
        Raises:
            SpeedMeasurementError: If measurement fails
            RepositoryNotFoundError: If repository not found
            EntryNotFoundError: If file not found
            HTTPError: If HTTP error occurs
            ConnectionError: If connection fails
            Timeout: If request times out
        """
        try:
            # Get download URL for sample file
            try:
                url = self.api.hf_hub_url(repo_id, sample_file, revision="main")
            except (RepositoryNotFoundError, EntryNotFoundError) as e:
                logger.error(f"Repository or file error: {e}")
                raise
            except Exception as e:
                logger.error(f"Error getting download URL: {e}")
                raise SpeedMeasurementError(f"Failed to get download URL: {e}")
            
            # Setup headers
            headers = {}
            if token:
                headers["authorization"] = f"Bearer {token}"
                
            # Measure download speed
            start_time = time.time()
            bytes_downloaded = 0
            
            try:
                with requests.get(url, headers=headers, stream=True) as response:
                    response.raise_for_status()
                    
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            bytes_downloaded += len(chunk)
                            current_time = time.time()
                            duration = current_time - start_time
                            
                            # Add measurement
                            try:
                                measurement = SpeedMeasurement(
                                    timestamp=current_time,
                                    bytes_transferred=bytes_downloaded,
                                    duration=duration
                                )
                                with self._lock:
                                    self._measurements.append(measurement)
                            except Exception as e:
                                logger.error(f"Error recording measurement: {e}")
                                raise SpeedMeasurementError(f"Failed to record measurement: {e}")
                                
                            # Stop after measure_duration seconds
                            if duration >= self.measure_duration:
                                break
                                
            except (HTTPError, ConnectionError, Timeout) as e:
                logger.error(f"Network error during measurement: {e}")
                raise
            except Exception as e:
                logger.error(f"Error during speed measurement: {e}")
                raise SpeedMeasurementError(f"Speed measurement failed: {e}")
                
            # Calculate average speed
            try:
                with self._lock:
                    speeds = [m.bytes_per_second for m in self._measurements]
                    if not speeds:
                        raise ValueError("No speed measurements collected")
                    avg_speed = mean(speeds)
                    
                    # Set allowed speed based on bandwidth percentage
                    self._allowed_speed = avg_speed * self.bandwidth_percentage
                    
                    logger.info(
                        f"Measured speed: {avg_speed:,.0f} B/s, "
                        f"Allowed speed: {self._allowed_speed:,.0f} B/s"
                    )
                    return self._allowed_speed
                    
            except ValueError as e:
                logger.error(f"Error calculating speed: {e}")
                raise SpeedMeasurementError(f"Speed calculation failed: {e}")
            except Exception as e:
                logger.error(f"Error processing measurements: {e}")
                raise SpeedMeasurementError(f"Failed to process measurements: {e}")
                
        except (RepositoryNotFoundError, EntryNotFoundError, HTTPError, 
                ConnectionError, Timeout):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during speed measurement: {e}")
            raise SpeedMeasurementError(f"Speed measurement failed: {e}")
            
    def allocate_thread_speed(
        self,
        thread_id: int,
        file_size: int,
        total_size: int,
        remaining_files: int,
        total_files: int
    ) -> float:
        """Calculate allowed speed for a specific thread
        
        Args:
            thread_id: Thread identifier
            file_size: Size of current file
            total_size: Total size of all files
            remaining_files: Number of files remaining
            total_files: Total number of files
            
        Returns:
            Allocated speed in bytes per second
            
        Raises:
            SpeedAllocationError: If allocation fails
        """
        try:
            if self._allowed_speed is None:
                raise ValueError("No speed measurement available")
                
            if file_size < 0 or total_size < 0:
                raise ValueError("File sizes cannot be negative")
                
            if remaining_files < 0 or total_files < 0:
                raise ValueError("File counts cannot be negative")
                
            if total_files < remaining_files:
                raise ValueError("Remaining files cannot exceed total files")
                
            with self._lock:
                # Calculate factors
                size_factor = file_size / total_size if total_size > 0 else 1
                count_factor = remaining_files / total_files if total_files > 0 else 1
                
                # Calculate thread speed
                thread_speed = (
                    self._allowed_speed * 
                    size_factor * 
                    (1 + count_factor)  # Boost factor based on remaining work
                )
                
                # Store allocated speed
                self._thread_speeds[thread_id] = thread_speed
                
                logger.debug(
                    f"Thread {thread_id} allocated speed: {thread_speed:,.0f} B/s"
                )
                return thread_speed
                
        except ValueError as e:
            logger.error(f"Invalid allocation parameters: {e}")
            raise SpeedAllocationError(f"Invalid parameters: {e}")
        except Exception as e:
            logger.error(f"Error allocating thread speed: {e}")
            raise SpeedAllocationError(f"Speed allocation failed: {e}")
            
    def get_thread_speed(self, thread_id: int) -> Optional[float]:
        """Get currently allocated speed for a thread
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Allocated speed in bytes per second or None if not allocated
            
        Raises:
            SpeedManagerError: If error accessing thread speed
        """
        try:
            with self._lock:
                return self._thread_speeds.get(thread_id)
        except Exception as e:
            logger.error(f"Error getting thread speed: {e}")
            raise SpeedManagerError(f"Failed to get thread speed: {e}")
            
    @property
    def allowed_speed(self) -> Optional[float]:
        """Get the current allowed speed (after bandwidth percentage)"""
        return self._allowed_speed