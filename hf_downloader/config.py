from dataclasses import dataclass
import multiprocessing
from typing import Union
from pydantic import validator, PositiveInt, confloat
from dataclasses import dataclass
from .config_mixins import NetworkConfigMixin, SecurityConfigMixin

@dataclass
class DownloadConfig:
    """Centralized configuration for download parameters with validation"""
    num_threads: int = 0
    chunk_size: PositiveInt = 1024 * 1024  # 1MB chunks
    min_free_space_mb: PositiveInt = 5000  # 5GB minimum free space
    file_size_threshold: PositiveInt = 200 * 1024 * 1024  # 200MB threshold
    min_speed_percentage: confloat(ge=1, le=100) = 5.0  # 1-100% range
    speed_test_duration: PositiveInt = 5  # seconds
    verify_downloads: bool = False
    fix_broken: bool = False
    force_download: bool = False
    max_retries: PositiveInt = 5
    connect_timeout: PositiveInt = 10  # seconds
    read_timeout: PositiveInt = 30  # seconds
    token_refresh_interval: PositiveInt = 3600  # 1 hour


    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_threads()
        self._convert_size_threshold()

    def _validate_threads(self):
        """Calculate optimal threads if auto-detected"""
        if isinstance(self.num_threads, str) and self.num_threads.lower() == 'auto':
            self.num_threads = self.calculate_optimal_threads()
        elif self.num_threads <= 0:
            self.num_threads = self.calculate_optimal_threads()

    def _convert_size_threshold(self):
        """Convert size threshold to bytes if needed"""
        if self.file_size_threshold < 1024:
            self.file_size_threshold *= 1024 * 1024

    @staticmethod
    def calculate_optimal_threads() -> int:
        """Calculate I/O-optimized thread count"""
        cpu_cores = multiprocessing.cpu_count()
        return min(32, max(8, cpu_cores * 4))

    @validator('min_speed_percentage')
    def validate_speed_percentage(cls, value):
        """Ensure speed percentage stays in valid range"""
        return max(1.0, min(100.0, value))