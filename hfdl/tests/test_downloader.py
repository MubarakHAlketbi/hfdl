import pytest
from pathlib import Path
from hfdl.downloader import HFDownloader

def test_normalize_repo_id():
    """Test repository ID normalization"""
    # Test with URL
    url = "https://huggingface.co/bert-base-uncased"
    assert HFDownloader._normalize_repo_id(url) == "bert-base-uncased"
    
    # Test with direct repo ID
    repo_id = "bert-base-uncased"
    assert HFDownloader._normalize_repo_id(repo_id) == "bert-base-uncased"
    
    # Test with trailing slash
    url_with_slash = "https://huggingface.co/bert-base-uncased/"
    assert HFDownloader._normalize_repo_id(url_with_slash) == "bert-base-uncased"

def test_normalize_repo_id_invalid():
    """Test repository ID normalization with invalid inputs"""
    # Test with empty input
    with pytest.raises(ValueError, match="Repository ID cannot be empty"):
        HFDownloader._normalize_repo_id("")
    
    # Test with invalid format (no slash)
    with pytest.raises(ValueError, match="Invalid repository ID format"):
        HFDownloader._normalize_repo_id("invalid-format")

def test_downloader_initialization():
    """Test downloader initialization with default values"""
    downloader = HFDownloader("username/repo-name")
    
    # Check default values
    assert str(downloader.download_dir) == str(Path("downloads"))
    assert downloader.repo_type == "model"
    assert downloader.resume == True
    assert downloader.enhanced_mode == False

def test_downloader_initialization_custom():
    """Test downloader initialization with custom values"""
    downloader = HFDownloader(
        model_id="username/repo-name",
        download_dir="custom_downloads",
        repo_type="dataset",
        verify=True,
        force=True,
        resume=False,
        enhanced_mode=True
    )
    
    # Check custom values
    assert str(downloader.download_dir) == str(Path("custom_downloads"))
    assert downloader.repo_type == "dataset"
    assert downloader.resume == False
    assert downloader.enhanced_mode == True

def test_thread_count_validation():
    """Test thread count validation"""
    # Test 'auto'
    assert HFDownloader(
        "username/repo-name",
        num_threads='auto'
    ).config.num_threads > 0
    
    # Test positive integer
    assert HFDownloader(
        "username/repo-name",
        num_threads=4
    ).config.num_threads == 4