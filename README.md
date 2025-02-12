# Hugging Face Downloader (hf_downloader.py)

A fast, reliable, and safe downloader for Hugging Face models and datasets. This tool provides better performance than git and huggingface_hub while ensuring safe downloads and intelligent resource management.

## Features

- 🚀 **Smart Downloads**: 
  * Guaranteed 5 MB/s minimum speed for large files
  * Adaptive thread management
  * Network-aware operations
  * Bandwidth optimization
  * Real-time speed statistics
- 💻 **Resource Aware**: 
  * Intelligent CPU thread management
  * Network bandwidth control
  * System load optimization
  * Maintains system responsiveness
  * Per-thread performance tracking
- 🔒 **Smart Authentication**:
  * Works with public repositories without token
  * Automatic token handling for private repos
  * Clear authentication guidance
  * Safe token management
  * Graceful auth failures
- 📁 **Organized Storage**: 
  * Automatic model-specific directory structure
  * Clean file organization
  * Resume capability
  * State persistence
  * Progress tracking
- 🛡️ **Safe Operations**: 
  * File integrity checks
  * Proper error handling
  * Network saturation prevention
  * Graceful interruption handling
  * Clean state management

## Requirements

- Python 3.7 or higher
- Minimum 2 CPU threads (recommended 4+ for optimal performance)
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
pip install requests huggingface_hub tqdm
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
```

### Command Line Options

- `-t, --threads`: Number of download threads (default: auto-detected)
- `-d, --directory`: Base download directory (default: "downloads")
- `-r, --repo_type`: Repository type ("model", "dataset", or "space")
- `--chunk-size`: Download chunk size in bytes
- `--min-free-space`: Minimum required free space in MB
- `--verify`: Verify existing downloads
- `--fix-broken`: Remove and redownload corrupted files
- `--force`: Force fresh download, ignore existing files

## Interrupt Handling

The tool provides robust interrupt handling and progress tracking:

1. **Safe Interruption**:
   - Press Ctrl+C to safely stop downloads
   - Current progress is saved automatically
   - Partial downloads are preserved
   - Clear status messages provided

2. **Progress Information**:
   ```
   Download speeds - Current: 25.5 MB/s, Average: 22.3 MB/s
   Thread usage - Active: 8, Average speed per thread: 3.2 MB/s
   ```

3. **Interrupt Status**:
   ```
   Download interrupted: 3/10 files completed (30.0%), 2 files in progress
   Active downloads being stopped:
   - model.safetensors: 1.2GB/4.8GB (25.0%)
   - config.json: 45KB/45KB (100.0%)
   ```

4. **Resume Support**:
   ```bash
   # Resume interrupted download
   python hf_downloader.py username/model_name
   # Will continue from where it left off
   ```

## Smart Download Management

The tool automatically adapts its behavior based on file size:

### Large Files (>200MB)
- Guarantees minimum 5 MB/s download speed
- Adjusts threads automatically for performance
- Shows real-time speed information
- Adapts to network conditions
- Provides detailed progress tracking

### Small Files (<200MB)
- Uses maximum available threads
- Optimizes for quick completion
- Prevents network saturation
- Maintains system responsiveness
- Efficient resource usage

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

## Features in Detail

### Intelligent Resource Management
- Automatically detects system capabilities
- Adapts to network conditions
- Prevents system overload
- Maintains responsiveness
- Optimizes bandwidth usage

### Safe Downloads
- Verifies file integrity
- Checks available disk space
- Supports download resume
- Handles network issues gracefully
- Provides clear error messages

### Progress Tracking
- Per-file progress bars
- Download speed information
- Thread adjustment notifications
- Clear status messages
- Performance statistics

## Examples

1. Download a public model:
```bash
python hf_downloader.py Qwen/Qwen2.5-Coder-32B-Instruct
```

2. Download a private model:
```bash
# Will automatically use token if needed
python hf_downloader.py username/private-model
```

3. Download with custom settings:
```bash
python hf_downloader.py username/model_name -t 6 -d models --verify
```

4. Resume interrupted download:
```bash
# Same command as before
python hf_downloader.py username/model_name
# Will continue from last saved state
```

## Error Handling

The tool provides clear error messages for common issues:
- Authentication requirements
- Network connectivity problems
- Insufficient disk space
- Invalid repository names
- Permission issues
- Download interruptions
- Speed-related issues

## Best Practices

1. **Authentication**:
   - Only login if needed for private repos
   - Keep token secure
   - Follow authentication prompts

2. **Thread Count**:
   - Let the tool auto-detect thread count
   - Tool will optimize based on file size
   - Maintains system responsiveness

3. **Disk Space**:
   - Ensure sufficient free space (default minimum: 1GB)
   - Tool will check before downloading
   - Clear error messages if space low

4. **Large Downloads**:
   - Tool automatically optimizes for files >200MB
   - Guaranteed minimum speed of 5 MB/s
   - Adaptive thread management

5. **Interruptions**:
   - Safe to interrupt with Ctrl+C
   - Progress is saved automatically
   - Downloads can be resumed
   - State is properly maintained

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Built using the Hugging Face Hub API
- Uses requests for efficient downloads
- Progress bars powered by tqdm