from pydantic import BaseModel, Field

class NetworkConfigMixin(BaseModel):
    """Network-related configuration settings"""
    connect_timeout: int = Field(default=10, gt=0)
    read_timeout: int = Field(default=30, gt=0)
    max_retries: int = Field(default=5, gt=0)

    class Config:
        validate_assignment = True

class SecurityConfigMixin(BaseModel):
    """Security-related configuration settings"""
    verify_ssl: bool = True
    checksum_verification: bool = True
    token_refresh_interval: int = Field(default=3600, ge=300)  # minimum 5 minutes

    class Config:
        validate_assignment = True