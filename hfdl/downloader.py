import os
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Union, Tuple
from huggingface_hub import (
    HfApi, 
    snapshot_download,
    get_token,
    hf_hub_download
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    LocalEntryNotFoundError
)
from tqdm import tqdm
from .config import DownloadConfig
from .thread_manager import ThreadManager
from .file_manager import FileManager, FileInfo
from .speed_manager import SpeedManager

logger = logging.getLogger(__name__)

class HFDownloadError(Exception):
    """Base exception for downloader errors"""
    pass

class HFDownloader:
    """Efficient downloader for Hugging Face models using official API methods"""
    
    def __init__(
        self,
        model_id: str,
        download_dir: str = "downloads",
        num_threads: Union[int, str] = 'auto',
        repo_type: str = "model",
        verify: bool = False,
        force: bool = False,
        resume: bool = True,
        size_threshold_mb: float = 100.0,
        bandwidth_percentage: float = 95.0,
        speed_measure_seconds: int = 8,
        enhanced_mode: bool = False
    ):
        """Initialize the downloader with configuration.
        
        Args:
            model_id (str): The model ID to download (e.g. "bert-base-uncased")
            download_dir (str): Directory to store downloads
            num_threads (Union[int, str]): Number of download threads or 'auto'
            repo_type (str): Repository type ("model", "dataset", or "space")
            verify (bool): Whether to verify downloads
            force (bool): Force fresh download
            resume (bool): Allow resuming partial downloads
            size_threshold_mb (float): Threshold in MB to categorize files
            bandwidth_percentage (float): Percentage of bandwidth to use
            speed_measure_seconds (int): Duration for speed measurement
            enhanced_mode (bool): Whether to use enhanced download features
        """
        self.model_id = self._normalize_repo_id(model_id)
        self.download_dir = Path(download_dir)
        self.repo_type = repo_type
        self.enhanced_mode = enhanced_mode
        
        # Convert 'auto' to 0 for thread count
        thread_count = 0 if isinstance(num_threads, str) and num_threads.lower() == 'auto' else num_threads
        
        # Initialize configuration
        self.config = DownloadConfig.create(
            num_threads=thread_count,
            verify_downloads=verify,
            force_download=force,
            size_threshold_mb=size_threshold_mb,
            bandwidth_percentage=bandwidth_percentage,
            speed_measure_seconds=speed_measure_seconds
        )
        self.resume = resume
        
        # Initialize HfApi instance once
        self.api = HfApi()
        self.token = self._get_auth_token()
        
        # Initialize managers for enhanced mode
        if enhanced_mode:
            self.thread_manager = ThreadManager()
            self.file_manager = FileManager(
                api=self.api,
                size_threshold_mb=self.config.size_threshold_mb
            )
            self.speed_manager = SpeedManager(
                api=self.api,
                measure_duration=self.config.speed_measure_seconds,
                bandwidth_percentage=self.config.bandwidth_percentage,
                chunk_size=self.config.download_chunk_size
            )
        else:
            self.thread_manager = None
            self.file_manager = None
            self.speed_manager = None

    @staticmethod
    def _normalize_repo_id(repo_id_or_url: str) -> str:
        """Normalize repository ID from URL or direct input."""
        if not repo_id_or_url:
            raise ValueError("Repository ID cannot be empty")
            
        # Remove URL prefix if present
        repo_id = repo_id_or_url.replace("https://huggingface.co/", "")
        repo_id = repo_id.rstrip("/")
        
        # Validate repository ID format
        if "/" not in repo_id:
            raise ValueError(
                f"Invalid repository ID format: {repo_id}\n"
                "Expected format: username/repository-name"
            )
            
        return repo_id

    def _get_auth_token(self) -> Optional[str]:
        """Get and validate authentication token."""
        try:
            # Get token using root method
            token = get_token()
            if not token:
                logger.warning("No authentication token found. Some repositories may be inaccessible.")
                logger.info("To authenticate, run: huggingface-cli login")
                return None

            # Validate token by making an API call
            self.api.whoami(token=token)
            return token
        except Exception as e:
            if "401" in str(e):
                logger.warning("Invalid or expired token. Please login again using: huggingface-cli login")
            else:
                logger.error(f"Error validating token: {e}")
            return None

    def _verify_repo_access(self) -> bool:
        """Verify repository exists and is accessible."""
        try:
            self.api.repo_info(
                repo_id=self.model_id,
                repo_type=self.repo_type,
                token=self.token
            )
            return True
        except RepositoryNotFoundError:
            logger.error(f"Repository not found: {self.model_id}")
            return False
        except Exception as e:
            logger.error(f"Error accessing repository: {e}")
            return False

    def _download_enhanced(self) -> bool:
        """Download using enhanced features with size-based optimization"""
        try:
            # Discover and categorize files
            small_files, big_files = self.file_manager.discover_files(
                repo_id=self.model_id,
                repo_type=self.repo_type,
                token=self.token
            )
            
            # Create output directory
            output_dir = self.download_dir / self.model_id.split('/')[-1]
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with self.thread_manager:
                # Download small files first
                if small_files:
                    logger.info(f"Downloading {len(small_files)} small files...")
                    for file in small_files:
                        if self.thread_manager.should_stop:
                            logger.info("Download cancelled by user")
                            return False
                            
                        try:
                            local_path = output_dir / file.local_path
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            hf_hub_download(
                                repo_id=self.model_id,
                                filename=file.path_in_repo,
                                repo_type=self.repo_type,
                                local_dir=output_dir,
                                local_dir_use_symlinks=False,
                                token=self.token,
                                force_download=self.config.force_download,
                                resume_download=self.resume
                            )
                            self.file_manager.update_progress(
                                file.path_in_repo,
                                file.size
                            )
                        except Exception as e:
                            logger.error(f"Error downloading {file.name}: {e}")
                            continue
                
                # Handle big files with bandwidth control
                if big_files:
                    logger.info(f"Downloading {len(big_files)} large files...")
                    
                    # Measure initial speed with first big file
                    try:
                        self.speed_manager.measure_initial_speed(
                            repo_id=self.model_id,
                            sample_file=big_files[0].path_in_repo,
                            token=self.token
                        )
                    except Exception as e:
                        logger.error(f"Speed measurement failed: {e}")
                        # Fall back to default download if speed measurement fails
                        return self._download_legacy()
                    
                    # Download big files with speed control
                    download_threads = self.thread_manager.get_download_threads()
                    total_size = sum(f.size for f in big_files)
                    
                    for i, file in enumerate(big_files):
                        if self.thread_manager.should_stop:
                            logger.info("Download cancelled by user")
                            return False
                            
                        try:
                            # Calculate thread speed allocation
                            thread_speed = self.speed_manager.allocate_thread_speed(
                                thread_id=i % download_threads,
                                file_size=file.size,
                                total_size=total_size,
                                remaining_files=len(big_files) - i,
                                total_files=len(big_files)
                            )
                            
                            local_path = output_dir / file.local_path
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Submit download to thread pool
                            self.thread_manager.submit_download(
                                hf_hub_download,
                                repo_id=self.model_id,
                                filename=file.path_in_repo,
                                repo_type=self.repo_type,
                                local_dir=output_dir,
                                local_dir_use_symlinks=False,
                                token=self.token,
                                force_download=self.config.force_download,
                                resume_download=self.resume
                            )
                        except Exception as e:
                            logger.error(f"Error downloading {file.name}: {e}")
                            continue
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced download failed: {e}")
            # Fall back to legacy download
            return self._download_legacy()

    def _download_legacy(self) -> bool:
        """Original download method using snapshot_download"""
        try:
            # Create output directory
            output_dir = self.download_dir / self.model_id.split('/')[-1]
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {self.model_id} to {output_dir}")

            try:
                # Use snapshot_download as root method
                snapshot_download(
                    repo_id=self.model_id,
                    repo_type=self.repo_type,
                    local_dir=output_dir,
                    token=self.token,
                    force_download=self.config.force_download,
                    resume_download=self.resume,
                    max_workers=self.config.num_threads,
                    tqdm_class=tqdm
                )
                
                logger.info(f"Successfully downloaded {self.model_id}")
                return True

            except (RepositoryNotFoundError, RevisionNotFoundError, LocalEntryNotFoundError) as e:
                logger.error(f"Download failed: {str(e)}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during download: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise HFDownloadError(str(e)) from e

    def download(self) -> bool:
        """Download the model using appropriate method based on configuration.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self._verify_repo_access():
            return False
            
        if self.enhanced_mode:
            return self._download_enhanced()
        else:
            return self._download_legacy()

def validate_threads(value):
    """Validate thread count, allowing 'auto' or positive integers."""
    if isinstance(value, str) and value.lower() == 'auto':
        return 0  # 0 means auto
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid thread count: {value}\n"
            "Must be 'auto' or a positive integer"
        )

def validate_percentage(value):
    """Validate percentage value between 0 and 100."""
    try:
        fvalue = float(value)
        if not 0 <= fvalue <= 100:
            raise ValueError
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid percentage: {value}\n"
            "Must be a number between 0 and 100"
        )

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Fast and reliable downloader for Hugging Face models and datasets."
    )
    parser.add_argument("repo_id_or_url", help="Repository ID or URL")
    parser.add_argument("-t", "--threads", type=validate_threads, default='auto',
                       help="Number of threads (auto or positive integer)")
    parser.add_argument("-d", "--directory", default="downloads",
                       help="Download directory")
    parser.add_argument("-r", "--repo_type", default="model",
                       choices=["model", "dataset", "space"],
                       help="Repository type")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloads")
    parser.add_argument("--force", action="store_true",
                       help="Force fresh download")
    parser.add_argument("--no-resume", action="store_true",
                       help="Disable download resuming")
    
    # Enhanced mode arguments
    parser.add_argument("--enhanced", action="store_true",
                       help="Enable enhanced download features")
    parser.add_argument("--size-threshold", type=float, default=100.0,
                       help="Size threshold in MB for file categorization")
    parser.add_argument("--bandwidth", type=validate_percentage, default=95.0,
                       help="Percentage of bandwidth to use")
    parser.add_argument("--measure-time", type=int, default=8,
                       help="Seconds to measure initial download speed")
    
    args = parser.parse_args()

    if not args.repo_id_or_url:
        args.repo_id_or_url = input("Enter the Hugging Face repository ID or URL: ")

    downloader = HFDownloader(
        model_id=args.repo_id_or_url,
        download_dir=args.directory,
        num_threads=args.threads,
        repo_type=args.repo_type,
        verify=args.verify,
        force=args.force,
        resume=not args.no_resume,
        size_threshold_mb=args.size_threshold,
        bandwidth_percentage=args.bandwidth,
        speed_measure_seconds=args.measure_time,
        enhanced_mode=args.enhanced
    )

    success = downloader.download()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
