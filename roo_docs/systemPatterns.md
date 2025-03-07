# System Patterns: HFDL - Hugging Face Download Library

## Architecture Overview
HFDL follows a modular architecture with clear separation of concerns:

1. **Core Downloader (HFDownloader)**: Main entry point that orchestrates the download process
2. **Thread Manager**: Handles thread allocation and control based on CPU availability
3. **File Manager**: Manages file operations, categorization, and tracking
4. **Speed Manager**: Controls bandwidth usage and optimizes download speeds
5. **Configuration**: Centralizes configuration parameters

## Key Technical Decisions

### 1. Dual Download Modes
- **Standard Mode**: Uses Hugging Face's built-in `snapshot_download` for simplicity
- **Enhanced Mode**: Uses custom implementation with file categorization and speed optimization

### 2. File Categorization
- Files are categorized as "small" or "large" based on a configurable size threshold
- Small files are downloaded sequentially to minimize overhead
- Large files are downloaded with thread and bandwidth optimization

### 3. Thread Management
- Thread allocation is based on CPU availability
- Different scenarios are handled based on CPU count:
  - Single-thread: All resources for download
  - Dual-thread: One for Ctrl+C handling, one for download
  - Triple-thread: One free, one for Ctrl+C, one for download
  - Multi-thread: One free, one for Ctrl+C, rest for download

### 4. Speed Control
- Initial speed measurement using a sample file
- Bandwidth usage percentage is configurable
- Thread speed allocation based on file size and remaining work

### 5. Error Handling
- Comprehensive exception hierarchy
- Automatic fallback to legacy mode on enhanced mode failure
- Graceful shutdown and resource cleanup

## Design Patterns

### 1. Manager Pattern
- Thread, File, and Speed managers encapsulate specific functionality
- Each manager has a clear responsibility and interface

### 2. Context Manager Pattern
- ThreadManager implements `__enter__` and `__exit__` for resource management
- Ensures proper initialization and cleanup

### 3. Factory Pattern
- DownloadConfig.create() factory method for configuration creation

### 4. Composition Over Inheritance
- Downloader composes managers rather than inheriting from them
- Promotes flexibility and maintainability

### 5. Data Transfer Objects
- FileInfo and SpeedMeasurement classes for structured data passing
- Encapsulates related data with validation