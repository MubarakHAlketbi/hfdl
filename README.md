# HFDL - Hugging Face Download Library (v. 0.3.1)

A fast and reliable downloader for Hugging Face models and datasets with intelligent optimization features that adapt to your system capabilities, network conditions, and specific needs.

<p align="center">
  <a href="https://pay.ziina.com/MubarakHAlketbi">
    <img src="https://img.shields.io/badge/Support_Me-Donate-9626ff?style=for-the-badge&logo=https%3A%2F%2Fimgur.com%2FvwC39JY" alt="Support Me - Donate">
  </a>
</p>

## Smart Systems

HFDL incorporates several intelligent systems that work together to optimize your download experience:

### 1. CPU-Based Thread Auto-Scaling
- **What it does**: Automatically determines the optimal number of threads based on your CPU cores
- **How it works**:
  - 1-2 CPU cores: Allocates 2 threads
  - 3-8 cores: Uses a number of threads equal to the core count
  - More than 8 cores: Caps at 8 threads to prevent overloading
- **Why it's smart**: Balances performance and resource usage without manual tuning

### 2. Size-Based File Categorization
- **What it does**: Classifies files as "small" or "big" based on a configurable threshold (default: 100 MB)
- **How it works**:
  - Small files: Downloaded quickly, often in parallel
  - Big files: Handled with bandwidth control for efficient resource allocation
- **Why it's smart**: Optimizes download strategy based on file characteristics

### 3. Bandwidth Measurement and Control
- **What it does**: Measures your download speed and limits usage to a percentage (default: 95%)
- **How it works**:
  - Measures initial speed with a sample file
  - Allocates bandwidth across threads for large files
  - Introduces micro-delays to maintain speed limits when needed
- **Why it's smart**: Prevents network saturation while maximizing throughput

### 4. Graceful Interruption Handling
- **What it does**: Ensures downloads can be safely interrupted without corrupted files
- **How it works**:
  - Uses a dedicated thread for interrupt signals on multi-core systems
  - Implements clean shutdown procedures for all resources
- **Why it's smart**: Provides reliability and responsiveness during long downloads

### 5. Comprehensive Error Handling
- **What it does**: Anticipates and manages a wide range of potential errors
- **How it works**:
  - Implements custom exception hierarchy for precise error handling
  - Provides fallback mechanisms and recovery strategies
- **Why it's smart**: Maintains operation even under adverse conditions

### 6. Progress Tracking
- **What it does**: Monitors and displays download progress at both file and overall levels
- **How it works**:
  - Tracks bytes downloaded for each file
  - Aggregates progress across all files for overall completion percentage
- **Why it's smart**: Provides real-time feedback with thread-safe accuracy

## Installation

```bash
pip install hfdl
```

Or install from source:
```bash
git clone https://github.com/yourusername/hfdl.git
cd hfdl
pip install -e .
```

## Quick Start

```python
from hfdl import HFDownloader

# Basic usage
downloader = HFDownloader("MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF")
downloader.download()

# Enhanced mode with custom settings
downloader = HFDownloader(
    "Anthropic/hh-rlhf",
    repo_type="dataset",
    enhanced_mode=True,
    size_threshold_mb=100,
    bandwidth_percentage=95
)
downloader.download()
```

## Command Line Usage

The CLI has been reorganized for better usability, with options grouped into Basic, Advanced, and Output categories.

### Interactive Mode

If you run `hfdl` without arguments, it will enter interactive mode and guide you through the process:

```bash
# Start interactive mode
hfdl
```

### Command Examples

```bash
# Basic usage
hfdl MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF

# Advanced mode with optimized downloading
hfdl Anthropic/hh-rlhf --optimize-download

# Custom threads and directory
hfdl Anthropic/hh-rlhf --threads 4 --directory ./models

# Test what would be downloaded without downloading
hfdl Anthropic/hh-rlhf --dry-run
```

### Available Options

```
Basic Options:
  -d, --directory DIR       Directory where files will be saved
  -r, --repo-type TYPE      Type of repository (model/dataset/space)
  --verify                  Verify integrity of downloaded files
  --force                   Force fresh download, overwriting existing files
  --no-resume               Disable download resuming

Advanced Options:
  --optimize-download       Enable optimized downloading with size-based
                            categorization and bandwidth control
  -t, --threads NUM         Number of download threads (auto: optimal based on
                            CPU cores, or specify a positive number)
  --size-threshold MB       Files larger than this size will use bandwidth control
  --bandwidth PERCENT       Percentage of measured bandwidth to use
  --measure-time SECS       Duration to measure initial download speed

Output Options:
  --quiet                   Suppress all output except errors
  --verbose                 Show detailed progress and debug information
  --dry-run                 Show what would be downloaded without downloading
```

## Cross-Platform Compatibility

HFDL is designed to work seamlessly across different operating systems:

- **Windows, macOS, and Linux** support
- **Path sanitization** to handle OS-specific filename restrictions
- **Adaptive file handling** that respects platform limitations

## Error Handling

HFDL provides comprehensive error handling and recovery mechanisms:

### Error Types

1. Download Errors:
   - `HFDownloadError`: Base exception for all errors
   - `ThreadManagerError`: Thread-related errors
   - `FileManagerError`: File operation errors
   - `SpeedManagerError`: Speed control errors

2. Specific Errors:
   - `FileSizeError`: File size calculation issues
   - `FileTrackingError`: Progress tracking issues
   - `SpeedMeasurementError`: Speed measurement issues
   - `SpeedAllocationError`: Speed allocation issues

### Error Recovery

HFDL implements automatic error recovery:
- Network retry mechanisms for transient failures
- Resource cleanup to prevent leaks
- State recovery to resume interrupted operations
- Fallback to legacy mode when enhanced features encounter issues
- OS-specific path handling to prevent filename-related errors

### Thread Safety

All operations are thread-safe:
- Resource protection with proper locking mechanisms
- State consistency across concurrent operations
- Safe cleanup even during interruptions
- Error propagation to the appropriate handlers
- Thread-aware progress tracking

## Testing

HFDL includes comprehensive test coverage:

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest hfdl/tests/

# Run specific test categories
pytest -v -k "error" hfdl/tests/      # Error handling tests
pytest -v -k "thread_safety" hfdl/tests/  # Thread safety tests
pytest -v hfdl/tests/test_downloader.py   # Downloader tests
```

### Test Categories

1. Unit Tests:
   - Component functionality
   - Error handling
   - Input validation
   - State management

2. Integration Tests:
   - Component interaction
   - Error propagation
   - Resource management
   - System behavior

3. Error Tests:
   - Error scenarios
   - Recovery mechanisms
   - Resource cleanup
   - State consistency

4. Thread Safety Tests:
   - Concurrent operations
   - Resource contention
   - State consistency
   - Error handling

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hfdl.git
cd hfdl
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest hfdl/tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.