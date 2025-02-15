# Test Catalog

This document catalogs all tests implemented in the HFDL test suite.

## Core Components

### Downloader Tests (test_downloader.py)

1. **Repository ID Tests**
   - `test_normalize_repo_id`: URL and ID normalization
     * Real model repo (MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF)
     * Real dataset repo (Anthropic/hh-rlhf)
     * URL variations
   
   - `test_normalize_repo_id_invalid`: Invalid input handling
     * Empty input
     * Invalid format

2. **Repository Access Tests**
   - `test_real_model_repo_access`: Real model access
   - `test_real_dataset_repo_access`: Real dataset access
   - `test_fake_repository_access`: Non-existent repo
   - `test_wrong_repo_type`: Invalid repo type

3. **Error Handling Tests**
   - `test_http_error_handling`: Network errors
   - `test_filesystem_error_handling`: File system errors
   - `test_environment_error_handling`: System errors
   - `test_entry_not_found_handling`: Missing files

### Thread Manager Tests (test_thread_manager.py)

1. **Initialization Tests**
   - `test_thread_scenario_detection`: CPU scenario detection
   - `test_initialization_error`: Init error handling
   - `test_signal_handler_error`: Signal handling errors
   - `test_thread_creation_error`: Thread creation errors

2. **Thread Pool Tests**
   - `test_thread_pool_error`: Pool creation errors
   - `test_submit_without_start`: Unstarted manager
   - `test_submit_error`: Task submission errors
   - `test_shutdown_error`: Shutdown errors

3. **Thread Safety Tests**
   - `test_ctrl_c_handler_error`: Signal handler errors
   - `test_context_manager`: Context handling
   - `test_graceful_shutdown`: Clean shutdown
   - `test_multiple_stop_calls`: Multiple stops

### File Manager Tests (test_file_manager.py)

1. **File Operation Tests**
   - `test_initialization_error`: Init error handling
   - `test_file_info_creation`: FileInfo creation
   - `test_repository_not_found`: Missing repo
   - `test_revision_not_found`: Bad revision

2. **Error Handling Tests**
   - `test_network_error`: Network failures
   - `test_metadata_error`: Metadata errors
   - `test_file_size_error`: Size calculation
   - `test_progress_tracking_error`: Progress errors

3. **Thread Safety Tests**
   - `test_progress_calculation_error`: Progress errors
   - `test_total_progress_error`: Total progress
   - `test_thread_safety`: Concurrent operations
   - `test_file_categorization`: Size categorization

### Speed Manager Tests (test_speed_manager.py)

1. **Initialization Tests**
   - `test_initialization_errors`: Init validation
   - `test_speed_measurement_calculation`: Speed calc
   - `test_repository_errors`: Repo errors
   - `test_network_errors`: Network errors

2. **Speed Measurement Tests**
   - `test_measurement_errors`: Measurement errors
   - `test_allocation_errors`: Allocation errors
   - `test_successful_speed_measurement`: Valid speed
   - `test_thread_speed_allocation`: Speed allocation

3. **Thread Safety Tests**
   - `test_thread_safety`: Concurrent operations
   - `test_multiple_thread_allocation`: Multi-thread
   - `test_speed_measurement_zero_duration`: Zero duration
   - `test_get_thread_speed`: Speed retrieval

## Test Categories

### Unit Tests
- Basic component functionality
- Input validation
- Error handling
- State management

### Integration Tests
- Component interaction
- Error propagation
- Resource management
- System behavior

### Error Tests
- Network errors
- File system errors
- Environment errors
- Missing files/entries

### Thread Safety Tests
- Concurrent operations
- Resource contention
- State consistency
- Error handling

## Running Tests

```bash
# Run all tests
pytest hfdl/tests/

# Run specific test file
pytest hfdl/tests/test_downloader.py

# Run error tests
pytest -v -k "error" hfdl/tests/

# Run thread safety tests
pytest -v -k "thread_safety" hfdl/tests/

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
```

## Shared Test Resources

### Fixtures (conftest.py)
- `mock_api`: Mock HfApi instance
- `temp_dir`: Temporary directory
- `real_model_repo`: Real model info
- `real_dataset_repo`: Real dataset info
- `fake_repos`: Fake repo info
- `sample_repo_files`: Sample files
- `sample_file_sizes`: Sample sizes
- `setup_logging`: Logging config

### Configuration (pytest.ini)
- Test markers
- Logging setup
- Test categories
- Output format

## Test Coverage Areas

1. **Error Handling**
   - Input validation
   - Network errors
   - File system errors
   - Thread errors
   - Resource errors

2. **Thread Safety**
   - Resource protection
   - State consistency
   - Error propagation
   - Clean shutdown

3. **Resource Management**
   - Initialization
   - Usage tracking
   - Cleanup
   - Error recovery

4. **Performance**
   - Speed measurement
   - Thread allocation
   - Resource usage
   - Error impact

## Future Test Areas

1. **Stress Testing**
   - High load scenarios
   - Resource limits
   - Error conditions
   - Recovery testing

2. **Performance Testing**
   - Speed benchmarks
   - Resource usage
   - Scalability tests
   - Optimization verification

3. **Network Testing**
   - Connection issues
   - Bandwidth limits
   - Timeout scenarios
   - Error recovery

4. **Integration Testing**
   - System integration
   - Component interaction
   - Error propagation
   - State management