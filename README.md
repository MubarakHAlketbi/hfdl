# HFDL - Hugging Face Download Manager (version 0.3.0)

An efficient downloader for Hugging Face models and datasets using official API methods.

## Key Features

- **Official API Integration**:
  - Uses huggingface_hub's snapshot_download
  - Built-in LFS support
  - Automatic cache management
  - Resume capability
  - Progress tracking

- **Resource Management**:
  - Optimized thread allocation
  - Built-in caching system
  - Automatic cleanup
  - System load awareness

- **Verification & Safety**:
  - Built-in file verification
  - Automatic integrity checks
  - Resume capability
  - Graceful interruption handling

- **Authentication**:
  - API-based token validation
  - Public repository support
  - Private repo access
  - Clear error messages

## Requirements

- Python 3.10+
- Required packages:
  ```python
  huggingface_hub >=0.28.1
  tqdm >=4.62.0
  pydantic >=2.0.0
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
    -d custom_dir \     # Custom directory (default: downloads)
    -t auto \           # Threads (auto or positive integer)
    -r model \          # Repository type (model, dataset, space)
    --verify \          # Verify downloads
    --force \           # Force fresh download
    --no-resume        # Disable resume capability
```

### Python API

```python
from hfdl import HFDownloader

downloader = HFDownloader(
    model_id="username/model_name",
    download_dir="custom_dir",     # default: "downloads"
    num_threads=0,                 # 0=auto, or positive integer
    repo_type="model",            # "model", "dataset", or "space"
    verify=False,                 # verify downloads
    force=False,                  # force fresh download
    resume=True                   # allow resume
)
downloader.download()
```

## Technical Implementation

### Core Components

1. **API Integration**:
   - Uses official huggingface_hub methods
   - Proper HfApi instance management
   - Built-in LFS support
   - Automatic caching

2. **Thread Management**:
   - Auto-scales based on system capabilities
   - Conservative thread allocation
   - Built-in optimization

3. **Download Management**:
   - Automatic resume capability
   - Progress tracking
   - Cache utilization
   - Error handling

4. **State Management**:
   - Managed by huggingface_hub
   - Automatic cache handling
   - Built-in progress tracking
   - File verification

### Directory Structure

```
downloads/
└── model-name/
    ├── config.json
    ├── model.safetensors
    └── pytorch_model.bin
```

## Best Practices

1. **Thread Management**:
   - Use 'auto' for optimal thread allocation
   - Or specify a positive integer for manual control
   - System will optimize based on available resources

2. **Download Options**:
   - Enable resume for reliable downloads
   - Use verify for extra safety
   - Force download when needed
   - Choose appropriate repo type

3. **Error Handling**:
   - Clear error messages
   - Proper validation
   - Automatic retry
   - Progress feedback

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

- Hugging Face Hub API for core functionality
- tqdm for progress bars