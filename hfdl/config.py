import multiprocessing
from typing import Union, Optional
from .validation import BaseConfig

class DownloadConfig(BaseConfig):
    """Configuration for download operations with validation and smart defaults"""
    
    @classmethod
    def create(cls, **kwargs) -> 'DownloadConfig':
        """Factory method to create config with proper validation"""
        # Handle 'auto' threads specification
        if isinstance(kwargs.get('num_threads'), str) and kwargs['num_threads'].lower() == 'auto':
            kwargs['num_threads'] = cls.calculate_optimal_threads()
        elif kwargs.get('num_threads', 0) <= 0:
            kwargs['num_threads'] = cls.calculate_optimal_threads()

        # Always convert file size threshold from MB to bytes
        if 'file_size_threshold' in kwargs:
            kwargs['file_size_threshold'] = kwargs['file_size_threshold'] * 1024 * 1024  # MB to bytes

        return cls(**kwargs)

    @staticmethod
    def calculate_optimal_threads() -> int:
        """Calculate I/O-optimized thread count based on system capabilities"""
        cpu_cores = multiprocessing.cpu_count()
        # More conservative thread allocation
        # For low-core systems (1-2 cores): use 2-4 threads
        # For medium systems (3-8 cores): use 2x cores
        # For high-core systems (>8 cores): use 3x cores up to max 32
        if cpu_cores <= 2:
            return max(2, cpu_cores * 2)
        elif cpu_cores <= 8:
            return cpu_cores * 2
        else:
            return min(32, cpu_cores * 3)

    def __str__(self) -> str:
        """Human-readable configuration representation"""
        return (
            f"DownloadConfig:\n"
            f"  Threads: {self.num_threads}\n"
            f"  Chunk size: {self.chunk_size / (1024*1024):.1f}MB\n"
            f"  Min free space: {self.min_free_space_mb}MB\n"
            f"  File size threshold: {self.file_size_threshold / (1024*1024):.1f}MB\n"
            f"  Min speed percentage: {self.min_speed_percentage}%\n"
            f"  Speed test duration: {self.speed_test_duration}s\n"
            f"  Verify downloads: {self.verify_downloads}\n"
            f"  Fix broken: {self.fix_broken}\n"
            f"  Force download: {self.force_download}\n"
            f"  Connect timeout: {self.connect_timeout}s\n"
            f"  Read timeout: {self.read_timeout}s\n"
            f"  Max retries: {self.max_retries}\n"
            f"  Token refresh interval: {self.token_refresh_interval}s"
        )