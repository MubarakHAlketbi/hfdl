from pydantic import BaseModel, Field
from typing import Optional
from .config_mixins import NetworkConfigMixin, SecurityConfigMixin

class BaseConfig(NetworkConfigMixin, SecurityConfigMixin):
    """Base validation model for configuration with comprehensive validation rules"""
    num_threads: Optional[int] = Field(default=0, ge=0)
    chunk_size: int = Field(default=1024 * 1024, ge=1024)  # 1MB default
    min_free_space_mb: int = Field(default=5000, ge=100)  # 5GB default
    file_size_threshold: int = Field(default=200 * 1024 * 1024, ge=0)  # 200MB default
    min_speed_percentage: float = Field(default=5.0, ge=1, le=100)
    speed_test_duration: int = Field(default=5, ge=1)
    verify_downloads: bool = False
    fix_broken: bool = False
    force_download: bool = False

    class Config:
        """Pydantic config"""
        validate_assignment = True
        extra = "forbid"

    @classmethod
    def create(cls, **kwargs):
        """Factory method to create config with validation"""
        if kwargs.get('file_size_threshold', 0) < 1024:
            kwargs['file_size_threshold'] *= 1024 * 1024
        return cls(**kwargs)