# Test Catalog

This document catalogs all tests implemented in the HFDL test suite.

## Downloader Tests (test_downloader.py)

### Repository ID Normalization
- `test_normalize_repo_id`: Validates repository ID normalization from URLs and direct IDs
  * Tests real model repo (MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF)
  * Tests real dataset repo (Anthropic/hh-rlhf)
  * Tests URLs with trailing slashes

- `test_normalize_repo_id_invalid`: Tests invalid repository ID handling
  * Empty input
  * Invalid format (no slash)

### Repository Initialization
- `test_real_model_initialization`: Tests model repository initialization
- `test_real_model_url_initialization`: Tests model URL initialization
- `test_real_dataset_initialization`: Tests dataset repository initialization
- `test_real_dataset_url_initialization`: Tests dataset URL initialization
- `test_downloader_initialization_custom`: Tests custom initialization parameters

### Repository Access
- `test_fake_repository_access`: Tests non-existent model repository
- `test_fake_dataset_access`: Tests non-existent dataset repository
- `test_real_model_repo_access`: Tests real model repository access (integration)
- `test_real_dataset_repo_access`: Tests real dataset repository access (integration)
- `test_nonexistent_repo_download`: Tests download of non-existent repository
- `test_wrong_repo_type`: Tests repository type validation
- `test_mixed_url_types`: Tests different URL format variations

### Error Handling
- `test_http_error_handling`: Tests network error handling
  * Connection failures
  * Timeout errors
  * Server errors

- `test_filesystem_error_handling`: Tests file system error handling
  * Permission denied
  * Disk full
  * Invalid paths

- `test_environment_error_handling`: Tests system environment errors
  * Resource limits
  * Memory constraints
  * System configuration issues

- `test_entry_not_found_handling`: Tests missing file/entry errors
  * Missing repository files
  * Invalid file paths
  * Non-existent entries

### Configuration
- `test_thread_count_validation`: Tests thread count validation
  * Auto thread detection
  * Manual thread specification

## Thread Manager Tests (test_thread_manager.py)

### Scenario Detection
- `test_thread_scenario_detection`: Tests CPU-based scenario selection
- `test_download_threads_calculation`: Tests download thread calculation

### Thread Management
- `test_thread_manager_context`: Tests context manager functionality
- `test_ctrl_c_handler`: Tests ctrl+c signal handling
- `test_submit_download`: Tests download task submission
- `test_submit_without_start`: Tests error handling for unstarted manager

## File Manager Tests (test_file_manager.py)

### File Information
- `test_file_info_creation`: Tests FileInfo dataclass creation
- `test_file_categorization`: Tests size-based file categorization

### Progress Tracking
- `test_progress_tracking`: Tests download progress tracking
- `test_total_progress`: Tests total progress calculation
- `test_thread_safety`: Tests thread-safe operations

### Error Handling
- `test_repository_not_found`: Tests repository not found error
- `test_thread_safety`: Tests concurrent operations

## Speed Manager Tests (test_speed_manager.py)

### Speed Measurement
- `test_speed_measurement_calculation`: Tests speed calculation
- `test_speed_measurement_zero_duration`: Tests zero duration handling
- `test_initial_speed_measurement`: Tests initial speed measurement

### Speed Allocation
- `test_thread_speed_allocation`: Tests speed allocation for threads
- `test_multiple_thread_allocation`: Tests multi-thread speed allocation
- `test_get_thread_speed`: Tests retrieving allocated speeds
- `test_get_nonexistent_thread_speed`: Tests non-existent thread handling

### Error Handling
- `test_speed_measurement_error_handling`: Tests measurement error handling
- `test_allocation_without_measurement`: Tests allocation without measurement
- `test_thread_safety`: Tests concurrent operations

## Shared Test Resources (conftest.py)

### Fixtures
- `mock_api`: Mock HfApi instance
- `temp_dir`: Temporary directory management
- `real_model_repo`: Real model repository information
- `real_dataset_repo`: Real dataset repository information
- `fake_repos`: Fake repository information
- `sample_repo_files`: Sample file list
- `sample_file_sizes`: Sample file sizes
- `setup_logging`: Logging configuration

### Configuration
- Test markers configuration
- Logging setup
- Cleanup handlers

## Test Categories

### Unit Tests
- Basic component functionality
- Input validation
- Error handling
- Configuration

### Integration Tests
- Repository access
- Download operations
- Thread management
- Speed control

### Error Tests
- Network errors
- File system errors
- Environment errors
- Missing file errors

## Running Tests

```bash
# Run all tests
pytest hfdl/tests/

# Run specific test file
pytest hfdl/tests/test_downloader.py

# Run integration tests only
pytest -m integration hfdl/tests/

# Run error handling tests
pytest -k "error" hfdl/tests/

# Run with detailed output
pytest -v hfdl/tests/
```

## Test Organization

```
hfdl/tests/
├── __init__.py
├── conftest.py        # Shared fixtures
├── pytest.ini        # Configuration
├── test_catalog.md   # This catalog
├── test_downloader.py
├── test_thread_manager.py
├── test_file_manager.py
└── test_speed_manager.py