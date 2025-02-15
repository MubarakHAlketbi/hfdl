from pydantic import BaseModel, Field
from typing import Optional, Literal

class BaseConfig(BaseModel):
    """Base validation model for configuration with comprehensive validation rules"""
    # Original parameters
    num_threads: Optional[int] = Field(default=0, ge=0)
    verify_downloads: bool = False
    force_download: bool = False
    repo_type: Literal["model", "dataset", "space"] = Field(default="model")
    download_dir: str = Field(default="downloads")
    
    # New parameters for enhanced download control
    size_threshold_mb: float = Field(
        default=100.0,
        ge=0.0,
        description="Threshold in MB to categorize files as small/big"
    )
    bandwidth_percentage: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of measured bandwidth to use"
    )
    speed_measure_seconds: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Duration in seconds for initial speed measurement"
    )
    download_chunk_size: int = Field(
        default=8192,
        ge=1024,
        le=1048576,  # 1MB max chunk size
        description="Chunk size in bytes for download operations"
    )

    class Config:
        """Pydantic config"""
        validate_assignment = True
        extra = "forbid"