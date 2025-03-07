import pytest
from pathlib import Path
import os
import sys
from unittest.mock import patch, Mock, MagicMock
from requests.exceptions import HTTPError
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    LocalEntryNotFoundError,
    EntryNotFoundError
)
from huggingface_hub.hf_api import RepoFile
from hfdl.downloader import HFDownloader
from hfdl.file_manager import FileInfo
from hfdl.cli import main

def test_normalize_repo_id(real_model_repo, real_dataset_repo):
    """Test repository ID normalization"""
    # Test model with URL
    assert HFDownloader._normalize_repo_id(real_model_repo['url']) == real_model_repo['id']
    
    # Test dataset with direct ID
    assert HFDownloader._normalize_repo_id(real_dataset_repo['id']) == real_dataset_repo['id']
    
    # Test with trailing slash
    url_with_slash = f"{real_model_repo['url']}/"
    assert HFDownloader._normalize_repo_id(url_with_slash) == real_model_repo['id']

def test_normalize_repo_id_invalid():
    """Test repository ID normalization with invalid inputs"""
    # Test with empty input
    with pytest.raises(ValueError, match="Repository ID cannot be empty"):
        HFDownloader._normalize_repo_id("")
    
    # Test with invalid format (no slash)
    with pytest.raises(ValueError, match="Invalid repository ID format"):
        HFDownloader._normalize_repo_id("invalid-format")

def test_http_error_handling(real_model_repo):
    """Test handling of HTTP errors"""
    downloader = HFDownloader(real_model_repo['id'])
    
    # Mock HTTP error in repo access
    with patch.object(downloader.api, 'repo_info', side_effect=HTTPError("Connection failed")):
        assert not downloader._verify_repo_access()
    
    # Mock HTTP error in download
    with patch('huggingface_hub.snapshot_download', side_effect=HTTPError("Download failed")):
        assert not downloader.download()

def test_filesystem_error_handling(real_model_repo, tmp_path):
    """Test handling of file system errors"""
    # Create a read-only directory
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    os.chmod(readonly_dir, 0o444)  # Read-only
    
    # Test with read-only directory
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(readonly_dir)
    )
    assert not downloader.download()
    
    # Cleanup
    os.chmod(readonly_dir, 0o777)

def test_environment_error_handling(real_model_repo):
    """Test handling of environment errors"""
    downloader = HFDownloader(real_model_repo['id'])
    
    # Mock environment error in directory creation
    with patch('pathlib.Path.mkdir', side_effect=EnvironmentError("Resource limit")):
        assert not downloader.download()

def test_entry_not_found_handling(real_model_repo):
    """Test handling of missing file/entry errors"""
    downloader = HFDownloader(real_model_repo['id'])
    
    # Mock entry not found in download
    with patch('huggingface_hub.snapshot_download', side_effect=EntryNotFoundError("File not found")):
        assert not downloader.download()

def test_real_model_initialization(real_model_repo):
    """Test downloader initialization with real model repository"""
    downloader = HFDownloader(
        real_model_repo['id'],
        repo_type=real_model_repo['type']
    )
    assert downloader.model_id == real_model_repo['id']
    assert downloader.repo_type == real_model_repo['type']

def test_real_model_url_initialization(real_model_repo):
    """Test downloader initialization with real model URL"""
    downloader = HFDownloader(
        real_model_repo['url'],
        repo_type=real_model_repo['type']
    )
    assert downloader.model_id == real_model_repo['id']
    assert downloader.repo_type == real_model_repo['type']

def test_real_dataset_initialization(real_dataset_repo):
    """Test downloader initialization with real dataset repository"""
    downloader = HFDownloader(
        real_dataset_repo['id'],
        repo_type=real_dataset_repo['type']
    )
    assert downloader.model_id == real_dataset_repo['id']
    assert downloader.repo_type == real_dataset_repo['type']

def test_real_dataset_url_initialization(real_dataset_repo):
    """Test downloader initialization with real dataset URL"""
    downloader = HFDownloader(
        real_dataset_repo['url'],
        repo_type=real_dataset_repo['type']
    )
    assert downloader.model_id == real_dataset_repo['id']
    assert downloader.repo_type == real_dataset_repo['type']

def test_fake_repository_access(fake_repos):
    """Test accessing a non-existent repository"""
    downloader = HFDownloader(
        fake_repos['model']['id'],
        repo_type=fake_repos['model']['type']
    )
    assert not downloader._verify_repo_access()

def test_fake_dataset_access(fake_repos):
    """Test accessing a non-existent dataset"""
    downloader = HFDownloader(
        fake_repos['dataset']['id'],
        repo_type=fake_repos['dataset']['type']
    )
    assert not downloader._verify_repo_access()

def test_downloader_initialization_custom(real_model_repo):
    """Test downloader initialization with custom values"""
    downloader = HFDownloader(
        model_id=real_model_repo['id'],
        download_dir="custom_downloads",
        repo_type=real_model_repo['type'],
        verify=True,
        force=True,
        resume=False,
        enhanced_mode=True
    )
    
    # Check custom values
    assert str(downloader.download_dir) == str(Path("custom_downloads"))
    assert downloader.repo_type == real_model_repo['type']
    assert downloader.resume == False
    assert downloader.enhanced_mode == True

def test_thread_count_validation(real_model_repo):
    """Test thread count validation"""
    # Test 'auto'
    assert HFDownloader(
        real_model_repo['id'],
        num_threads='auto'
    ).config.num_threads > 0
    
    # Test positive integer
    assert HFDownloader(
        real_model_repo['id'],
        num_threads=4
    ).config.num_threads == 4

@pytest.mark.integration
def test_real_model_repo_access(real_model_repo):
    """Test accessing a real model repository"""
    downloader = HFDownloader(
        real_model_repo['id'],
        repo_type=real_model_repo['type']
    )
    assert downloader._verify_repo_access()

@pytest.mark.integration
def test_real_dataset_repo_access(real_dataset_repo):
    """Test accessing a real dataset repository"""
    downloader = HFDownloader(
        real_dataset_repo['id'],
        repo_type=real_dataset_repo['type']
    )
    assert downloader._verify_repo_access()

@pytest.mark.integration
def test_nonexistent_repo_download(fake_repos):
    """Test attempting to download a non-existent repository"""
    downloader = HFDownloader(
        fake_repos['model']['id'],
        repo_type=fake_repos['model']['type']
    )
    assert not downloader.download()

@pytest.mark.integration
def test_wrong_repo_type(real_model_repo, real_dataset_repo):
    """Test using wrong repository type"""
    # Try to download dataset as model
    downloader = HFDownloader(
        real_dataset_repo['id'],
        repo_type="model"  # Wrong type, should be dataset
    )
    assert not downloader._verify_repo_access()

    # Try to download model as dataset
    downloader = HFDownloader(
        real_model_repo['id'],
        repo_type="dataset"  # Wrong type, should be model
    )
    assert not downloader._verify_repo_access()

@pytest.mark.integration
def test_mixed_url_types(real_model_repo, real_dataset_repo):
    """Test different URL formats"""
    # Test model with and without dataset prefix
    model_with_prefix = f"https://huggingface.co/datasets/{real_model_repo['id']}"
    model_without_prefix = real_model_repo['url']
    
    # Both should normalize to the same ID
    assert HFDownloader._normalize_repo_id(model_with_prefix) == real_model_repo['id']
    assert HFDownloader._normalize_repo_id(model_without_prefix) == real_model_repo['id']
    
    # Test dataset with and without dataset prefix
    dataset_with_prefix = real_dataset_repo['url']
    dataset_without_prefix = f"https://huggingface.co/{real_dataset_repo['id']}"
    
    # Both should normalize to the same ID
    assert HFDownloader._normalize_repo_id(dataset_with_prefix) == real_dataset_repo['id']
    assert HFDownloader._normalize_repo_id(dataset_without_prefix) == real_dataset_repo['id']

@pytest.mark.integration
def test_enhanced_mode_download(real_model_repo, tmp_path):
    """Test enhanced mode download with mocked components"""
    # Create downloader with enhanced mode
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(tmp_path),
        enhanced_mode=True
    )
    
    # Mock file discovery
    small_file = FileInfo(
        name="small.txt",
        size=50 * 1024 * 1024,  # 50MB
        path_in_repo="small.txt",
        local_path=Path("small.txt")
    )
    
    big_file = FileInfo(
        name="big.txt",
        size=200 * 1024 * 1024,  # 200MB
        path_in_repo="big.txt",
        local_path=Path("big.txt")
    )
    
    # Mock file manager's discover_files method
    with patch.object(downloader.file_manager, 'discover_files', return_value=([small_file], [big_file])):
        # Mock speed measurement
        with patch.object(downloader.speed_manager, 'measure_initial_speed', return_value=1000000):
            # Mock thread manager's submit_download method
            mock_future = MagicMock()
            mock_future.result.return_value = "downloaded_file.txt"
            
            with patch.object(downloader.thread_manager, 'submit_download', return_value=mock_future):
                # Mock hf_hub_download for small files
                with patch('huggingface_hub.hf_hub_download', return_value="small_file.txt"):
                    # Test download
                    assert downloader.download()

@pytest.mark.integration
def test_enhanced_mode_no_big_files(real_model_repo, tmp_path):
    """Test enhanced mode download with no big files"""
    # Create downloader with enhanced mode
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(tmp_path),
        enhanced_mode=True
    )
    
    # Mock file discovery with only small files
    small_file = FileInfo(
        name="small.txt",
        size=50 * 1024 * 1024,  # 50MB
        path_in_repo="small.txt",
        local_path=Path("small.txt")
    )
    
    # Mock file manager's discover_files method
    with patch.object(downloader.file_manager, 'discover_files', return_value=([small_file], [])):
        # Mock hf_hub_download for small files
        with patch('huggingface_hub.hf_hub_download', return_value="small_file.txt"):
            # Test download
            assert downloader.download()

@pytest.mark.integration
def test_enhanced_mode_download_failure(real_model_repo, tmp_path):
    """Test enhanced mode download with failures"""
    # Create downloader with enhanced mode
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(tmp_path),
        enhanced_mode=True
    )
    
    # Mock file discovery
    small_file = FileInfo(
        name="small.txt",
        size=50 * 1024 * 1024,  # 50MB
        path_in_repo="small.txt",
        local_path=Path("small.txt")
    )
    
    big_file = FileInfo(
        name="big.txt",
        size=200 * 1024 * 1024,  # 200MB
        path_in_repo="big.txt",
        local_path=Path("big.txt")
    )
    
    # Mock file manager's discover_files method
    with patch.object(downloader.file_manager, 'discover_files', return_value=([small_file], [big_file])):
        # Mock speed measurement
        with patch.object(downloader.speed_manager, 'measure_initial_speed', return_value=1000000):
            # Mock thread manager's submit_download method to return a failed future
            mock_future = MagicMock()
            mock_future.result.return_value = None  # Failed download
            
            with patch.object(downloader.thread_manager, 'submit_download', return_value=mock_future):
                # Mock hf_hub_download to fail for small files
                with patch('huggingface_hub.hf_hub_download', side_effect=HTTPError("Download failed")):
                    # Test download should fail
                    assert not downloader.download()

@pytest.mark.integration
def test_resume_download(real_model_repo, tmp_path):
    """Test resuming downloads"""
    # Create downloader with resume enabled
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(tmp_path),
        resume=True
    )
    
    # Mock snapshot_download to verify resume parameter
    with patch('huggingface_hub.snapshot_download') as mock_snapshot:
        mock_snapshot.return_value = str(tmp_path)
        downloader.download()
        
        # Verify resume parameter was passed
        mock_snapshot.assert_called_once()
        assert mock_snapshot.call_args.kwargs['resume_download'] == True

@pytest.mark.integration
def test_force_download(real_model_repo, tmp_path):
    """Test forcing downloads"""
    # Create downloader with force enabled
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(tmp_path),
        force=True
    )
    
    # Mock snapshot_download to verify force parameter
    with patch('huggingface_hub.snapshot_download') as mock_snapshot:
        mock_snapshot.return_value = str(tmp_path)
        downloader.download()
        
        # Verify force parameter was passed
        mock_snapshot.assert_called_once()
        assert mock_snapshot.call_args.kwargs['force_download'] == True

@pytest.mark.integration
def test_verify_download(real_model_repo, tmp_path):
    """Test verifying downloads"""
    # Create downloader with verify enabled
    downloader = HFDownloader(
        real_model_repo['id'],
        download_dir=str(tmp_path),
        verify=True
    )
    
    # Verify that config has verify_downloads set to True
    assert downloader.config.verify_downloads == True

@pytest.mark.integration
def test_cli_basic(real_model_repo, tmp_path):
    """Test basic CLI functionality"""
    # Mock sys.argv
    test_args = [
        'hfdl',
        real_model_repo['id'],
        '-d', str(tmp_path),
        '-r', real_model_repo['type']
    ]
    
    with patch('sys.argv', test_args):
        # Mock HFDownloader to avoid actual download
        with patch('hfdl.cli.HFDownloader') as mock_downloader:
            # Mock download method to return True
            mock_instance = Mock()
            mock_instance.download.return_value = True
            mock_downloader.return_value = mock_instance
            
            # Run CLI
            with patch('sys.exit') as mock_exit:
                main()
                
                # Verify downloader was created with correct args
                mock_downloader.assert_called_once()
                args, kwargs = mock_downloader.call_args
                assert kwargs['model_id'] == real_model_repo['id']
                assert kwargs['download_dir'] == str(tmp_path)
                assert kwargs['repo_type'] == real_model_repo['type']
                
                # Verify download was called
                mock_instance.download.assert_called_once()
                
                # Verify exit code
                mock_exit.assert_called_once_with(0)

@pytest.mark.integration
def test_cli_enhanced_mode(real_model_repo, tmp_path):
    """Test CLI with enhanced mode"""
    # Mock sys.argv
    test_args = [
        'hfdl',
        real_model_repo['id'],
        '-d', str(tmp_path),
        '--enhanced',
        '--size-threshold', '50',
        '--bandwidth', '80',
        '--measure-time', '5'
    ]
    
    with patch('sys.argv', test_args):
        # Mock HFDownloader to avoid actual download
        with patch('hfdl.cli.HFDownloader') as mock_downloader:
            # Mock download method to return True
            mock_instance = Mock()
            mock_instance.download.return_value = True
            mock_downloader.return_value = mock_instance
            
            # Run CLI
            with patch('sys.exit') as mock_exit:
                main()
                
                # Verify downloader was created with correct args
                mock_downloader.assert_called_once()
                args, kwargs = mock_downloader.call_args
                assert kwargs['model_id'] == real_model_repo['id']
                assert kwargs['download_dir'] == str(tmp_path)
                assert kwargs['enhanced_mode'] == True
                assert kwargs['size_threshold_mb'] == 50.0
                assert kwargs['bandwidth_percentage'] == 80.0
                assert kwargs['speed_measure_seconds'] == 5
                
                # Verify download was called
                mock_instance.download.assert_called_once()
                
                # Verify exit code
                mock_exit.assert_called_once_with(0)

@pytest.mark.integration
def test_cli_download_failure(real_model_repo, tmp_path):
    """Test CLI with download failure"""
    # Mock sys.argv
    test_args = [
        'hfdl',
        real_model_repo['id'],
        '-d', str(tmp_path)
    ]
    
    with patch('sys.argv', test_args):
        # Mock HFDownloader to simulate download failure
        with patch('hfdl.cli.HFDownloader') as mock_downloader:
            # Mock download method to return False
            mock_instance = Mock()
            mock_instance.download.return_value = False
            mock_downloader.return_value = mock_instance
            
            # Run CLI
            with patch('sys.exit') as mock_exit:
                main()
                
                # Verify download was called
                mock_instance.download.assert_called_once()
                
                # Verify exit code
                mock_exit.assert_called_once_with(1)