from pydantic import BaseModel, Field
from typing import Optional, Literal

class BaseConfig(BaseModel):
    """Base validation model for configuration with comprehensive validation rules"""
    num_threads: Optional[int] = Field(default=0, ge=0)
    verify_downloads: bool = False
    force_download: bool = False
    repo_type: Literal["model", "dataset", "space"] = Field(default="model")
    download_dir: str = Field(default="downloads")

    class Config:
        """Pydantic config"""
        validate_assignment = True
        extra = "forbid"