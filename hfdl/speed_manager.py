import time
import threading
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from statistics import mean
import requests
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

@dataclass
class SpeedMeasurement:
    """Represents a speed measurement sample"""
    timestamp: float
    bytes_transferred: int
    duration: float
    
    @property
    def bytes_per_second(self) -> float:
        """Calculate bytes per second for this measurement"""
        return self.bytes_transferred / self.duration if self.duration > 0 else 0

class SpeedManager:
    """Manages download speed measurement and control
    
    Handles:
    - Initial speed measurement
    - Bandwidth allocation
    - Speed tracking per thread
    - Thread-safe operations
    """
    
    def __init__(
        self,
        api: HfApi,
        measure_duration: int,
        bandwidth_percentage: float,
        chunk_size: int
    ):
        self.api = api
        self.measure_duration = measure_duration
        self.bandwidth_percentage = bandwidth_percentage / 100.0
        self.chunk_size = chunk_size
        
        self._measurements: List[SpeedMeasurement] = []
        self._thread_speeds: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._allowed_speed: Optional[float] = None
        
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
            ValueError: If measurement fails
        """
        try:
            # Get download URL for sample file
            url = self.api.hf_hub_url(repo_id, sample_file, revision="main")
            
            # Setup headers
            headers = {}
            if token:
                headers["authorization"] = f"Bearer {token}"
                
            # Measure download speed
            start_time = time.time()
            bytes_downloaded = 0
            
            with requests.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        bytes_downloaded += len(chunk)
                        current_time = time.time()
                        duration = current_time - start_time
                        
                        # Add measurement
                        measurement = SpeedMeasurement(
                            timestamp=current_time,
                            bytes_transferred=bytes_downloaded,
                            duration=duration
                        )
                        with self._lock:
                            self._measurements.append(measurement)
                            
                        # Stop after measure_duration seconds
                        if duration >= self.measure_duration:
                            break
                            
            # Calculate average speed
            with self._lock:
                speeds = [m.bytes_per_second for m in self._measurements]
                avg_speed = mean(speeds) if speeds else 0
                
            # Set allowed speed based on bandwidth percentage
            self._allowed_speed = avg_speed * self.bandwidth_percentage
            
            logger.info(
                f"Measured speed: {avg_speed:,.0f} B/s, "
                f"Allowed speed: {self._allowed_speed:,.0f} B/s"
            )
            return self._allowed_speed
            
        except Exception as e:
            logger.error(f"Failed to measure speed: {e}")
            raise ValueError(f"Speed measurement failed: {e}")
            
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
            ValueError: If no speed measurement available
        """
        if self._allowed_speed is None:
            raise ValueError("No speed measurement available")
            
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
            
    def get_thread_speed(self, thread_id: int) -> Optional[float]:
        """Get currently allocated speed for a thread
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Allocated speed in bytes per second or None if not allocated
        """
        with self._lock:
            return self._thread_speeds.get(thread_id)
            
    @property
    def allowed_speed(self) -> Optional[float]:
        """Get the current allowed speed (after bandwidth percentage)"""
        return self._allowed_speed