# Product Context: HFDL - Hugging Face Download Library

## Why This Project Exists
HFDL (Hugging Face Download Library) was created to provide a fast and reliable way to download models and datasets from Hugging Face. It addresses common issues with downloading large AI models and datasets, such as slow speeds, interrupted downloads, and inefficient resource utilization.

## Problems It Solves
1. **Inefficient Downloads**: Standard downloaders don't optimize for different file sizes or network conditions
2. **Resource Utilization**: Poor CPU and bandwidth utilization during downloads
3. **Interrupted Downloads**: Lack of robust error handling and recovery mechanisms
4. **Progress Tracking**: Limited visibility into download progress
5. **Speed Control**: No intelligent bandwidth management

## How It Should Work
HFDL should provide a seamless experience for downloading Hugging Face resources with the following workflow:

1. **Initialization**: User provides a repository ID and optional configuration
2. **Repository Verification**: System verifies the repository exists and is accessible
3. **File Discovery**: System categorizes files based on size (small vs. large)
4. **Speed Measurement**: For enhanced mode, system measures available bandwidth
5. **Optimized Download**:
   - Small files are downloaded sequentially
   - Large files are downloaded with thread and bandwidth optimization
6. **Progress Tracking**: User receives real-time progress updates
7. **Error Handling**: System handles errors gracefully with automatic recovery
8. **Completion**: All files are verified and available locally

The library should be usable both as a Python module and as a command-line tool, with sensible defaults but extensive customization options.