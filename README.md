# HFDL - Hugging Face Download Library (v. 0.3.1)

A fast and reliable downloader for Hugging Face models and datasets with enhanced features.

## Features

- Smart file categorization based on size
- CPU-based thread auto-scaling
- Bandwidth control and optimization
- Progress tracking
- Comprehensive error handling
- Extensive test coverage

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

```bash
# Basic usage
hfdl MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF

# Enhanced mode
hfdl Anthropic/hh-rlhf --enhanced --size-threshold 100 --bandwidth 95

# Full options
hfdl [repo_id] [options]
  --enhanced            Enable enhanced features
  --size-threshold MB   Size threshold for file categorization
  --bandwidth PERCENT   Bandwidth usage percentage
  --measure-time SECS   Speed measurement duration
  --threads NUM        Number of download threads
  --directory PATH     Download directory
  --repo-type TYPE     Repository type (model/dataset/space)
  --verify             Verify downloads
  --force              Force fresh download
  --no-resume          Disable download resuming
```

## Error Handling

HFDL provides comprehensive error handling:

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
- Network retry mechanisms
- Resource cleanup
- State recovery
- Fallback to legacy mode

### Thread Safety

All operations are thread-safe:
- Resource protection
- State consistency
- Safe cleanup
- Error propagation

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