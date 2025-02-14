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

        # Convert file size threshold if needed
        if kwargs.get('file_size_threshold', 0) < 1024:
            kwargs['file_size_threshold'] *= 1024 * 1024

        return cls(**kwargs)

    @staticmethod
    def calculate_optimal_threads() -> int:
        """Calculate I/O-optimized thread count based on system capabilities"""
        cpu_cores = multiprocessing.cpu_count()
        # Use 4x CPU cores, but keep within reasonable bounds
        # Minimum 8 threads for I/O operations
        # Maximum 32 threads to prevent resource exhaustion
        return min(32, max(8, cpu_cores * 4))

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