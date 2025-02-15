import multiprocessing
from typing import Union, Optional
from .validation import BaseConfig

class DownloadConfig(BaseConfig):
    """Configuration for download operations with validation and smart defaults
    
    Handles both basic download settings and enhanced features:
    - Thread management and optimization
    - File size categorization
    - Bandwidth control
    - Speed measurement
    """
    
    @classmethod
    def create(cls, **kwargs) -> 'DownloadConfig':
        """Factory method to create config with proper validation
        
        Handles:
        - Automatic thread calculation when num_threads <= 0
        - Validation of all configuration parameters
        - Smart defaults for enhanced features
        """
        # Handle auto thread specification (0 means auto)
        if kwargs.get('num_threads', 0) <= 0:
            kwargs['num_threads'] = cls.calculate_optimal_threads()

        return cls(**kwargs)

    @staticmethod
    def calculate_optimal_threads() -> int:
        """Calculate I/O-optimized thread count based on system capabilities
        
        Thread allocation strategy:
        - For low-core systems (1-2 cores): use 2 threads
        - For medium systems (3-8 cores): use core count
        - For high-core systems (>8 cores): use 8 threads max
        """
        cpu_cores = multiprocessing.cpu_count()
        # Conservative thread allocation:
        # For low-core systems (1-2 cores): use 2 threads
        # For medium systems (3-8 cores): use core count
        # For high-core systems (>8 cores): use 8 threads max
        if cpu_cores <= 2:
            return 2
        elif cpu_cores <= 8:
            return cpu_cores
        else:
            return 8

    def __str__(self) -> str:
        """Human-readable configuration representation"""
        return (
            f"DownloadConfig:\n"
            f"  Threads: {self.num_threads}\n"
            f"  Verify downloads: {self.verify_downloads}\n"
            f"  Force download: {self.force_download}\n"
            f"  Download directory: {self.download_dir}\n"
            f"  Repository type: {self.repo_type}\n"
            f"  Size threshold (MB): {self.size_threshold_mb}\n"
            f"  Bandwidth usage (%): {self.bandwidth_percentage}\n"
            f"  Speed measure time (s): {self.speed_measure_seconds}\n"
            f"  Download chunk size: {self.download_chunk_size:,} bytes"
        )