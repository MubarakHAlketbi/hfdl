class NetworkConfigMixin:
    connect_timeout: int = 10
    read_timeout: int = 30
    max_retries: int = 5

class SecurityConfigMixin:
    verify_ssl: bool = True
    checksum_verification: bool = True
    token_refresh_interval: int = 3600