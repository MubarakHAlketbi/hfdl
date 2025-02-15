import pytest
from pathlib import Path
import os
from unittest.mock import patch, Mock
from requests.exceptions import HTTPError
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    LocalEntryNotFoundError,
    EntryNotFoundError
)
from hfdl.downloader import HFDownloader

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