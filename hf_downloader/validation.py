from pydantic import BaseModel, conint, confloat
from typing import Optional

class BaseConfig(BaseModel):
    """Base validation model for configuration"""
    num_threads: Optional[conint(ge=0)] = 0
    chunk_size: conint(ge=1024) = 1024 * 1024
    min_free_space_mb: conint(ge=100) = 5000
    file_size_threshold: conint(ge=0) = 200 * 1024 * 1024
    min_speed_percentage: confloat(ge=1, le=100) = 5.0
    speed_test_duration: conint(ge=1) = 5