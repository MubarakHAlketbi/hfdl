# HFDL - Hugging Face Download Manager (version 0.1.10)

An optimized downloader for Hugging Face models and datasets with intelligent resource management and verification.

## Key Features

- **Efficient Download Strategies**:
  - Size-based prioritization (small files first)
  - Configurable file size threshold (default: 200MB)
  - Fallback size retrieval for API limitations
  - Verified resume capability with Range requests

- **Resource Management**:
  - CPU-optimized thread allocation
  - Network bandwidth monitoring
  - 5% disk space safety buffer
  - Cross-platform file locking
  - System load awareness

- **Verification & Safety**:
  - Hybrid hashing (BLAKE3 for speed + SHA256 compatibility)
  - File integrity checks
  - Pre-allocation of disk space
  - Graceful interruption handling
  - Corrupted file redownload

- **Authentication**:
  - Public repository support
  - Private repo token handling
  - Token validation
  - Clear authentication errors

## Requirements

- Python 3.10+
- Required packages:
  ```python
  requests >=2.26.0
  huggingface_hub >=0.11.0
  tqdm >=4.62.0
  portalocker >=2.3.2
  blake3 >=0.3.0
  pydantic >=2.0.0
  typing-extensions >=4.0.0
  ```

## Installation

```bash
pip install hfdl
```

From source:
```bash
git clone https://github.com/MubarakHAlketbi/hfdl.git
cd hfdl
pip install -e .
```

Authentication (for private repos):
```bash
huggingface-cli login
```

## Usage

### Command Line

```bash
# Basic download
hfdl username/model_name

# Advanced options
hfdl username/model_name \
    -d custom_dir \          # Custom directory (default: downloads)
    -t 8 \                   # Threads (0=auto)
    --min-free-space 5000 \  # Minimum free space in MB (default)
    --verify \               # Verify existing files
    --fix-broken \           # Redownload corrupted files
    --force                  # Ignore existing files
```

### Python API

```python
from hfdl import HFDownloader

downloader = HFDownloader(
    model_id="username/model_name",
    download_dir="custom_dir",
    num_threads=8,
    min_free_space=5000,      # MB
    file_size_threshold=200,  # MB
    verify=True,
    fix_broken=True,
    force=False
)
downloader.download()
```

## Technical Implementation

### Core Components

1. **File Classification**:
   - Separates files into small/big based on threshold
   - Handles missing size information via HEAD requests

2. **Thread Management**:
   - Auto-scales based on CPU cores:
     - 1-2 cores: 2-4 threads
     - 3-8 cores: 2x cores
     - 8+ cores: min(32, 3x cores)
   - Separate strategies for small/big files

3. **Speed Control**:
   - Initial speed test for big files
   - Dynamic rate limiting
   - Minimum 1MB/s per thread
   - Rolling average speed tracking

4. **State Management**:
   - JSON state file with schema validation
   - Progress tracking per file
   - Automatic resume capability
   - File lock synchronization

### Verification System

1. **Hybrid Hashing**:
   - BLAKE3 for fast large file verification
   - SHA256 for API compatibility
   - Size validation fallback

2. **Safety Checks**:
   - 5% disk space buffer
   - Pre-allocation with posix_fallocate/seek
   - Network timeout handling
   - Retry with exponential backoff

## Directory Structure

```
downloads/
└── model-name/
    ├── config.json
    ├── model.safetensors
    ├── pytorch_model.bin
    └── .download_state  # JSON progress tracking
```

## Best Practices

1. **Network Considerations**:
   - Use default auto-threading unless specific needs
   - Higher --file-size-threshold for slow connections
   - Monitor with --speed-check-interval

2. **System Resources**:
   - Allow 1GB+ buffer for min-free-space
   - Prefer fewer threads for low-memory systems
   - SSD benefits from higher thread counts

3. **Resuming Downloads**:
   - Interrupt with Ctrl+C preserves state
   - --verify checks existing files
   - --fix-broken removes corrupted data

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Linting
flake8 hfdl/
mypy hfdl/
```

## License

MIT - See [LICENSE](LICENSE)

## Acknowledgments

- Hugging Face Hub API
- Requests for HTTP handling
- tqdm progress bars
- Portalocker for file locking