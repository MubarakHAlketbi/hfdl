# HFDL - Hugging Face Download Manager (v0.1.1)

A fast, reliable, and safe downloader for Hugging Face models and datasets. This tool provides better performance than git and huggingface_hub while ensuring safe downloads and intelligent resource management.

## Features

- 🚀 **Smart Downloads**:
  * Size-based download strategy (small files first)
  * Dynamic thread allocation based on speed
  * Percentage-based speed allocation per thread
  * Real-time speed statistics
  * Progress tracking
  * Verified resume capability
  * Fallback size retrieval
- 💻 **Resource Aware**: 
  * I/O optimized thread management
  * Reserved thread for system responsiveness
  * Network bandwidth control
  * System load optimization
  * Per-thread performance tracking
  * Cross-platform file locking
- 🔒 **Smart Authentication**:
  * Works with public repositories without token
  * Secure token handling for private repos
  * Automatic token refresh
  * Clear authentication guidance
  * Safe token management
  * Graceful auth failures
- 📁 **Organized Storage**: 
  * Automatic model-specific directory structure
  * Clean file organization
  * Verified resume capability
  * State persistence with schema validation
  * Progress tracking
  * Race condition prevention
- 🛡️ **Safe Operations**: 
  * Hybrid hashing (BLAKE3 for speed, SHA256 for compatibility)
  * Pre-allocated file space
  * 5% disk space safety buffer
  * Network saturation prevention
  * Graceful interruption handling
  * Clean state management

## Requirements

- Python 3.10 or higher
- Required packages:
  * requests
  * huggingface_hub
  * tqdm
  * portalocker
  * blake3
- Sufficient disk space for downloads
- Network connectivity
- Hugging Face token (only for private repositories)

## Installation

You can install the package directly from PyPI:

```bash
pip install hfdl
```

Or install from source:

```bash
git clone https://github.com/MubarakHAlketbi/hfdl.git
cd hfdl
pip install -e .
```

Authentication (Optional):
```bash
huggingface-cli login
# Follow prompts to enter your token
```

Requirements:
- Python 3.10 or higher
- Network connectivity
- Sufficient disk space for downloads

## Usage

### Usage

1. Command Line Interface:
```bash
# Basic usage
hfdl username/model_name

# Advanced options
hfdl username/model_name \
    -t 8 \                          # Number of threads
    -d custom_dir \                 # Custom download directory
    --min-free-space 2000 \        # Minimum free space in MB
    --verify \                      # Verify downloads
    --fix-broken \                  # Fix broken files
    --file-size-threshold 500       # Size threshold in MB
```

2. Python API:
```python
from hfdl import HFDownloader

# Basic usage
downloader = HFDownloader("username/model_name")
downloader.download()

# Advanced configuration
downloader = HFDownloader(
    model_id="username/model_name",
    download_dir="custom_dir",
    num_threads=8,
    min_free_space=2000,
    verify=True,
    fix_broken=True,
    file_size_threshold=500
)
downloader.download()
```

### Command Line Options

- `-t, --threads`: Number of download threads (default: auto-detected, I/O optimized)
- `-d, --directory`: Base download directory (default: "downloads")
- `-r, --repo_type`: Repository type ("model", "dataset", or "space")
- `--chunk-size`: Download chunk size in bytes
- `--min-free-space`: Minimum required free space in MB
- `--verify`: Verify existing downloads (includes checksum validation)
- `--fix-broken`: Remove and redownload corrupted files
- `--force`: Force fresh download, ignore existing files
- `--file-size-threshold`: Size threshold for big files in MB (default: 200)
- `--min-speed-percentage`: Target minimum speed per thread as percentage of average speed (1-100, default: 5)
- `--speed-test-duration`: Duration of speed test in seconds (default: 5)
- `--speed-check-interval`: Interval for speed checks and rate limiting updates in seconds (default: 5)

## Download Strategy

The tool implements an intelligent size-based download strategy:

1. **File Classification**:
   - Small files (<200MB by default)
   - Big files (>200MB by default)
   - Configurable size threshold
   - Fallback size retrieval for API issues

2. **Download Priority**:
   - Small files downloaded first using all threads
   - Big files processed after small files complete
   - Ensures efficient resource utilization
   - Handles missing file sizes gracefully

3. **Speed Optimization**:
   - Initial speed test for first big file
   - Dynamic thread allocation based on speed
   - Percentage-based speed allocation per thread
   - Minimum 1 MB/s per thread safeguard
   - Adaptive performance optimization

4. **Progress Tracking**:
   - Real-time speed statistics
   - Per-file progress monitoring
   - Thread utilization tracking
   - Clear status updates
   - Improved error reporting

## Thread Management
The tool uses an intelligent thread allocation strategy optimized for different system sizes:

1. **Small Files**:
   - Uses optimal thread count based on system capabilities
   - Conservative allocation for low-core systems (1-2 cores: 2-4 threads)
   - Balanced allocation for medium systems (3-8 cores: 2x cores)
   - Scaled allocation for high-core systems (>8 cores: 3x cores, max 32)
   - Quick completion of small files

2. **Big Files**:
   - Initial single-thread speed test
   - Dynamic thread allocation based on speed
   - Each thread gets configurable percentage of total speed
   - Minimum speed safeguards (3 MB/s per thread)
   - Adaptive performance optimization

3. **System Resources**:
   - Smart thread scaling based on CPU cores
   - Network-aware thread management
   - Dynamic performance adjustment
   - Resource monitoring with safety limits
   - Cross-platform file locking

## Speed Management

The tool implements an advanced speed optimization system:

1. **Speed Testing**:
   - Initial speed test for big files
   - Average speed calculation
   - Thread optimization based on speed
   - Percentage-based speed allocation
   - Edge case handling

2. **Thread Allocation**:
   - Dynamic thread count for big files
   - Based on speed test results
   - Each thread gets X% of total speed (configurable)
   - Minimum speed safeguards
   - Ensures optimal performance

3. **Progress Tracking**:
   - Real-time speed statistics
   - Active thread monitoring
   - Overall progress tracking
   - Clear status updates
   - Better error reporting

## State Management

The tool provides robust state management:

1. **State Validation**:
   - Schema validation for state files
   - Corruption detection and recovery
   - Safe state persistence
   - Clear error reporting
   - Automatic recovery mechanisms

2. **Token Management**:
   - Automatic token refresh
   - Configurable refresh intervals
   - Safe token handling
   - Clear authentication errors
   - Graceful token updates

3. **File Integrity**:
   - Hybrid hashing (BLAKE3 for large files)
   - SHA256 compatibility with API
   - Pre-allocated file space
   - 5% disk space safety buffer
   - Clear integrity reporting

## Directory Structure

Downloads are organized by model name:
```
downloads/
├── model1_name/
│   ├── config.json
│   ├── model.safetensors
│   ├── .download_state    # Progress tracking
│   └── ...
└── model2_name/
    ├── config.json
    ├── model.safetensors
    ├── .download_state    # Progress tracking
    └── ...
```

## Best Practices

1. **Download Strategy**:
   - Let small files download first
   - Allow speed testing for big files
   - Use default thread optimization
   - Monitor download progress
   - Handle edge cases gracefully

2. **Configuration**:
   - Adjust file size threshold based on network
   - Set min-speed-percentage based on stability needs
   - Allow dynamic thread allocation
   - Monitor performance metrics
   - Use appropriate timeouts

3. **System Resources**:
   - Tool manages threads optimally for I/O
   - Prevents system overload
   - Maintains responsiveness
   - Enables safe interruption
   - Uses cross-platform file locking

4. **Interruptions**:
   - Safe to interrupt with Ctrl+C
   - Progress is saved automatically
   - Downloads can be resumed safely
   - State is properly maintained
   - Clear error reporting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- Built using the Hugging Face Hub API
- Uses requests for efficient downloads
- Progress bars powered by tqdm
- Cross-platform file locking by portalocker
- Fast hashing by BLAKE3
