# Hugging Face Downloader (hf_downloader.py)

A fast, reliable, and safe downloader for Hugging Face models and datasets. This tool provides better performance than git and huggingface_hub while ensuring safe downloads and intelligent resource management.

## Features

- 🚀 **Smart Downloads**:
  * Size-based download strategy (small files first)
  * Dynamic thread allocation for big files
  * Speed testing for optimal performance
  * Real-time speed statistics
  * Progress tracking
  * Verified resume capability
  * Fallback size retrieval for API issues
- 💻 **Resource Aware**: 
  * I/O optimized thread management
  * Reserved thread for system responsiveness
  * Network bandwidth control
  * System load optimization
  * Per-thread performance tracking
  * Directory creation locks
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
  * Comprehensive file integrity checks
  * Multiple checksum validation methods
  * Network saturation prevention
  * Graceful interruption handling
  * Clean state management
  * Edge case handling

## Requirements

- Python 3.7 or higher
- Required packages:
  * requests
  * huggingface_hub
  * tqdm
- Sufficient disk space for downloads
- Network connectivity
- Hugging Face token (only for private repositories)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up authentication:
```bash
huggingface-cli login
# Follow prompts to enter your token
```

## Usage

### Basic Usage

```bash
# Download public model (no authentication needed)
python hf_downloader.py username/model_name

# Download private model (will use token if needed)
python hf_downloader.py username/private-model
```

### Advanced Usage

```bash
# Custom configuration
python hf_downloader.py username/model_name -t 8 -d custom_dir --min-free-space 2000

# Verify and fix downloads
python hf_downloader.py username/model_name --verify --fix-broken

# Performance tuning
python hf_downloader.py username/model_name --file-size-threshold 500 --min-speed-per-thread 5
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
- `--min-speed-per-thread`: Minimum speed per thread in MB/s (default: 3)
- `--speed-test-duration`: Duration of speed test in seconds (default: 5)

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
   - Maintains minimum speed per thread
   - Adaptive performance optimization
   - Edge case handling for speed calculations

4. **Progress Tracking**:
   - Real-time speed statistics
   - Per-file progress monitoring
   - Thread utilization tracking
   - Clear status updates
   - Improved error reporting

## Thread Management

The tool uses an I/O-optimized thread allocation strategy:

1. **Small Files**:
   - Uses optimal thread count for I/O operations
   - Maximum parallel downloads
   - Quick completion of small files
   - Clear I/O vs CPU thread distinction

2. **Big Files**:
   - Initial single-thread speed test
   - Dynamic thread allocation based on speed
   - Ensures minimum speed per thread
   - Adaptive performance optimization
   - Improved thread messaging

3. **System Resources**:
   - I/O-optimized thread scaling
   - Network-aware thread management
   - Dynamic performance adjustment
   - Resource monitoring
   - Directory operation locks

## Speed Management

The tool implements an advanced speed optimization system:

1. **Speed Testing**:
   - Initial speed test for big files
   - Average speed calculation
   - Thread optimization based on speed
   - Minimum speed guarantees
   - Edge case handling

2. **Thread Allocation**:
   - Dynamic thread count for big files
   - Based on speed test results
   - Maintains minimum speed per thread
   - Ensures optimal performance
   - Improved error handling

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
   - Multiple checksum verification methods
   - Fallback to model index SHA
   - Size verification
   - Partial file validation
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
   - Set minimum speed based on connection
   - Allow dynamic thread allocation
   - Monitor performance metrics
   - Use appropriate timeouts

3. **System Resources**:
   - Tool manages threads optimally for I/O
   - Prevents system overload
   - Maintains responsiveness
   - Enables safe interruption
   - Handles directory permissions

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
