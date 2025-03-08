import os
import logging
import argparse
import time
import requests
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any
from . import __version__
import platform
from requests.exceptions import HTTPError
from huggingface_hub import (
    HfApi,
    snapshot_download,
    get_token,
    hf_hub_download
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    LocalEntryNotFoundError,
    EntryNotFoundError
)
from tqdm import tqdm
from .config import DownloadConfig
from .thread_manager import ThreadManager
from .file_manager import FileManager, FileInfo
from .speed_manager import SpeedManager
from .utils import sanitize_filename, get_os_compatible_path

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
        """Initialize the downloader with configuration."""
        try:
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
                
        except OSError as e:
            logger.error(f"File system error during initialization: {e}")
            raise HFDownloadError(f"File system error: {e}") from e
        except EnvironmentError as e:
            logger.error(f"Environment error during initialization: {e}")
            raise HFDownloadError(f"System environment error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise HFDownloadError(str(e)) from e

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
        except HTTPError as e:
            if "401" in str(e):
                logger.warning("Invalid or expired token. Please login again using: huggingface-cli login")
            else:
                logger.error(f"Network error validating token: {e}")
            return None
        except Exception as e:
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
        except HTTPError as e:
            logger.error(f"Network error accessing repository: {e}")
            return False
        except Exception as e:
            logger.error(f"Error accessing repository: {e}")
            return False
            
    def _speed_controlled_download(
        self,
        repo_id: str,
        filename: str,
        local_dir: Union[str, Path],
        thread_id: int,
        max_speed_bps: float,
        **kwargs
    ) -> Optional[str]:
        """
        Download a file with speed control.
        
        Args:
            repo_id: Repository ID
            filename: Path to file in repository
            local_dir: Local directory to save file
            thread_id: Thread ID for speed allocation
            max_speed_bps: Maximum speed in bytes per second
            **kwargs: Additional arguments for download
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Get download URL
            url = self.api.hf_hub_url(repo_id, filename, revision="main")
            
            # Setup headers
            headers = {}
            if self.token:
                headers["authorization"] = f"Bearer {self.token}"
                
            # Prepare local path
            local_dir = Path(local_dir)
            # Sanitize the filename to be OS-compatible
            safe_filename = get_os_compatible_path(filename)
            local_path = local_dir / safe_filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file info for progress tracking
            etag = None
            temp_file = None
            resume_size = 0
            
            # Check if file exists for resuming
            if local_path.exists() and kwargs.get("resume_download", True):
                temp_file = str(local_path)
                resume_size = os.path.getsize(temp_file)
                if resume_size > 0:
                    headers["Range"] = f"bytes={resume_size}-"
                    logger.info(f"Resuming download from {resume_size} bytes")
            else:
                if kwargs.get("force_download", False) and local_path.exists():
                    logger.info(f"Removing existing file: {local_path}")
                    local_path.unlink()
                    
                # Create temporary file
                temp_file = str(local_path) + ".partial"
                resume_size = 0
                
            # Open file for writing
            mode = "ab" if resume_size > 0 else "wb"
            
            with open(temp_file, mode) as f:
                with requests.get(url, headers=headers, stream=True) as response:
                    response.raise_for_status()
                    
                    # Get total file size
                    total_size = int(response.headers.get("content-length", 0))
                    if resume_size > 0:
                        total_size += resume_size
                        
                    # Setup progress tracking
                    progress = tqdm(
                        unit="B",
                        unit_scale=True,
                        total=total_size,
                        initial=resume_size,
                        desc=f"Downloading {filename}"
                    )
                    
                    # Download with speed control
                    chunk_size = self.config.download_chunk_size
                    downloaded = resume_size
                    start_time = time.time()
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(len(chunk))
                            
                            # Update file manager progress
                            if self.file_manager:
                                self.file_manager.update_progress(filename, downloaded)
                                
                            # Apply speed control if needed
                            if max_speed_bps > 0:
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    current_speed = downloaded / elapsed
                                    if current_speed > max_speed_bps:
                                        # Calculate sleep time to maintain speed limit
                                        sleep_time = (downloaded / max_speed_bps) - elapsed
                                        if sleep_time > 0:
                                            time.sleep(sleep_time)
                    
                    progress.close()
                    
            # Rename temp file to final file if needed
            if temp_file != str(local_path):
                os.replace(temp_file, local_path)
                
            logger.info(f"Successfully downloaded {filename}")
            return str(local_path)
            
        except HTTPError as e:
            logger.error(f"HTTP error downloading {filename}: {e}")
            return None
        except OSError as e:
            logger.error(f"OS error downloading {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None

    def _download_enhanced(self) -> bool:
        """Download using enhanced features with size-based optimization"""
        try:
            # Verify download directory
            try:
                self.download_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Cannot create download directory: {e}")
                raise HFDownloadError(f"File system error: Cannot create download directory - {e}")
            except EnvironmentError as e:
                logger.error(f"System error creating directory: {e}")
                raise HFDownloadError(f"System error: Cannot create directory - {e}")
            
            # Discover and categorize files
            try:
                small_files, big_files = self.file_manager.discover_files(
                    repo_id=self.model_id,
                    repo_type=self.repo_type,
                    token=self.token
                )
            except EntryNotFoundError as e:
                logger.error(f"Repository content not found: {e}")
                return False
            except HTTPError as e:
                logger.error(f"Network error discovering files: {e}")
                return False
            
            # Create output directory
            # Use sanitized model name for directory
            model_dirname = sanitize_filename(self.model_id.split('/')[-1])
            output_dir = self.download_dir / model_dirname
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Track failed downloads
            failed_files = []
            
            with self.thread_manager:
                # Download small files first
                if small_files:
                    logger.info(f"Downloading {len(small_files)} small files...")
                    
                    # Use thread pool for small files too for better performance
                    small_file_futures = []
                    for file in small_files:
                        if self.thread_manager.should_stop:
                            logger.info("Download cancelled by user")
                            return False
                            
                        try:
                            local_path = output_dir / file.local_path
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Submit to thread pool
                            future = self.thread_manager.submit_download(
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
                            small_file_futures.append((future, file))
                            
                        except Exception as e:
                            logger.error(f"Error submitting download for {file.name}: {e}")
                            failed_files.append(file.path_in_repo)
                    
                    # Check results of small file downloads
                    for future, file in small_file_futures:
                        try:
                            result = future.result()
                            if result:
                                self.file_manager.update_progress(
                                    file.path_in_repo,
                                    file.size
                                )
                            else:
                                failed_files.append(file.path_in_repo)
                                logger.error(f"Failed to download: {file.name}")
                        except Exception as e:
                            failed_files.append(file.path_in_repo)
                            logger.error(f"Error downloading {file.name}: {e}")
                
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
                    except HTTPError as e:
                        logger.error(f"Network error measuring speed: {e}")
                        return self._download_legacy()
                    except Exception as e:
                        logger.error(f"Speed measurement failed: {e}")
                        return self._download_legacy()
                else:
                    logger.info("No large files detected; downloading small files without speed control.")
                    
                    # Download big files with speed control
                    download_threads = self.thread_manager.get_download_threads()
                    total_size = sum(f.size for f in big_files)
                    big_file_futures = []
                    
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
                            
                            # Submit download to thread pool with speed control
                            future = self.thread_manager.submit_download(
                                self._speed_controlled_download,
                                repo_id=self.model_id,
                                filename=file.path_in_repo,
                                local_dir=output_dir,
                                thread_id=i % download_threads,
                                max_speed_bps=thread_speed,
                                repo_type=self.repo_type,
                                force_download=self.config.force_download,
                                resume_download=self.resume
                            )
                            big_file_futures.append((future, file))
                            
                        except Exception as e:
                            logger.error(f"Error submitting download for {file.name}: {e}")
                            failed_files.append(file.path_in_repo)
                    
                    # Check results of big file downloads
                    for future, file in big_file_futures:
                        try:
                            result = future.result()
                            if not result:
                                failed_files.append(file.path_in_repo)
                                logger.error(f"Failed to download: {file.name}")
                        except Exception as e:
                            failed_files.append(file.path_in_repo)
                            logger.error(f"Error downloading {file.name}: {e}")
            
            # Check for failed downloads
            if failed_files:
                for file_path in failed_files:
                    logger.error(f"Failed to download: {file_path}")
                return False
                
            return True
            
        except EnvironmentError as e:
            logger.error(f"System environment error: {e}")
            return self._download_legacy()
        except Exception as e:
            logger.error(f"Enhanced download failed: {e}")
            return self._download_legacy()

    def _download_legacy(self) -> bool:
        """Original download method using snapshot_download"""
        try:
            # Create output directory
            try:
                # Use sanitized model name for directory
                model_dirname = sanitize_filename(self.model_id.split('/')[-1])
                output_dir = self.download_dir / model_dirname
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Cannot create output directory: {e}")
                raise HFDownloadError(f"File system error: Cannot create directory - {e}")
            except EnvironmentError as e:
                logger.error(f"System error creating directory: {e}")
                raise HFDownloadError(f"System error: Cannot create directory - {e}")

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
            except EntryNotFoundError as e:
                logger.error(f"Repository content not found: {e}")
                return False
            except HTTPError as e:
                logger.error(f"Network error during download: {e}")
                return False
            except OSError as e:
                logger.error(f"File system error during download: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during download: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise HFDownloadError(str(e)) from e

    def download(self) -> bool:
        """Download the model using appropriate method based on configuration."""
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

def get_example_text():
    """Return formatted examples for the help text."""
    examples = [
        "Basic download: hfdl MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF",
        "Advanced mode: hfdl Anthropic/hh-rlhf --optimize-download",
        "Custom threads: hfdl Anthropic/hh-rlhf --threads 4",
        "Force download: hfdl Anthropic/hh-rlhf --directory ./models --force",
        "Dataset download: hfdl Anthropic/hh-rlhf --repo-type dataset"
    ]
    return "\n  ".join(["Examples:"] + examples)

def main():
    """Command line interface with user-friendly options."""
    # Create the main parser with a more detailed description
    parser = argparse.ArgumentParser(
        description="Fast and reliable downloader for Hugging Face models and datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_example_text()
    )
    # Add version argument
    parser.add_argument('-v', '--version',
                       action='version',
                       version=f'%(prog)s {__version__}')
    
    # Create argument groups for better organization
    basic_group = parser.add_argument_group('Basic Options', 'Essential options for most download needs')
    advanced_group = parser.add_argument_group('Advanced Options', 'Additional options for fine-tuning download behavior')
    output_group = parser.add_argument_group('Output Options', 'Control the verbosity and behavior of the tool')
    
    # Model ID argument (with nargs='?' to make it optional for interactive mode)
    parser.add_argument("repo_id_or_url", nargs='?',
                       help="Repository ID or URL (format: username/repo-name)")
    
    # Basic options
    basic_group.add_argument("-d", "--directory", default="downloads",
                       help="Directory where files will be saved (default: ./downloads)")
    basic_group.add_argument("-r", "--repo-type", default="model",
                       choices=["model", "dataset", "space"],
                       help="Type of repository to download (default: model)")
    basic_group.add_argument("--verify", action="store_true",
                       help="Verify integrity of downloaded files")
    basic_group.add_argument("--force", action="store_true",
                       help="Force fresh download, overwriting existing files")
    basic_group.add_argument("--no-resume", action="store_true",
                       help="Start download from beginning, disabling resume capability")
    
    # Advanced options (renamed and improved descriptions)
    advanced_group.add_argument("--optimize-download", action="store_true", dest="enhanced",
                       help="Enable optimized download with size-based categorization and bandwidth control")
    advanced_group.add_argument("-t", "--threads", type=validate_threads, default='auto',
                       help="Number of download threads (auto: optimal based on CPU cores, or specify a positive number)")
    advanced_group.add_argument("--size-threshold", type=float, default=100.0,
                       help="Files larger than this size (in MB) will use bandwidth control (default: 100.0)")
    advanced_group.add_argument("--bandwidth", type=validate_percentage, default=95.0,
                       help="Percentage of measured bandwidth to use for downloading large files (default: 95.0)")
    advanced_group.add_argument("--measure-time", type=int, default=8,
                       help="Duration in seconds to measure initial download speed (default: 8)")
    
    # Output control options
    output_group.add_argument("--quiet", action="store_true",
                       help="Suppress all output except errors")
    output_group.add_argument("--verbose", action="store_true",
                       help="Show detailed progress and debug information")
    output_group.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without actually downloading files")
    args = parser.parse_args()

    # Interactive mode if no repository is specified
    if not args.repo_id_or_url:
        print("Welcome to the Hugging Face Download Library (HFDL) interactive mode!")
        args.repo_id_or_url = input("\nEnter the model or dataset ID (e.g., bert-base-uncased): ")
        
        # Only ask for directory if not specified
        if args.directory == "downloads":
            custom_dir = input(f"Enter download directory (default: ./downloads): ")
            if custom_dir:
                args.directory = custom_dir
        
        # Ask about optimization
        if not args.enhanced:
            optimize = input("Enable optimized downloading features? (y/n, default: n): ").lower()
            args.enhanced = optimize.startswith('y')
    
    # Handle dry run mode
    if args.dry_run:
        print(f"\nDRY RUN: Would download {args.repo_id_or_url} as a {args.repo_type}")
        print(f"Download directory: {args.directory}")
        print(f"Optimized download: {'Enabled' if args.enhanced else 'Disabled'}")
        if args.enhanced:
            print(f"Thread count: {args.threads}")
            print(f"Size threshold: {args.size_threshold} MB")
            print(f"Bandwidth usage: {args.bandwidth}%")
        print("\nNo files will be downloaded in dry-run mode.")
        return 0


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

    # Configure logging based on verbosity flags
    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # If quiet mode is enabled, suppress stdout print statements
    if args.quiet:
        sys.stdout = open(os.devnull, 'w')
    
    try:
        success = downloader.download()
        
        # Restore stdout if it was redirected
        if args.quiet:
            sys.stdout = sys.__stdout__
            
        if success:
            print("\nDownload completed successfully")
        else:
            print("\nDownload completed with errors", file=sys.stderr)
            return 1
        return 0
    except Exception as e:
        # Restore stdout if it was redirected
        if args.quiet:
            sys.stdout = sys.__stdout__
        print(f"\nError: {str(e)}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
