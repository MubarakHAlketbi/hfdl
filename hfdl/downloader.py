import os
import logging
from pathlib import Path
from typing import Optional, List, Union
from huggingface_hub import (
    HfApi, 
    snapshot_download,
    get_token
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    LocalEntryNotFoundError
)
from tqdm import tqdm
from .config import DownloadConfig

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
        resume: bool = True
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
        """
        self.model_id = self._normalize_repo_id(model_id)
        self.download_dir = Path(download_dir)
        self.repo_type = repo_type
        self.config = DownloadConfig.create(
            num_threads=num_threads,
            verify_downloads=verify,
            force_download=force
        )
        self.resume = resume
        
        # Initialize HfApi instance once
        self.api = HfApi()
        self.token = self._get_auth_token()

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

    def download(self) -> bool:
        """Download the model using official Hugging Face Hub methods.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            if not self._verify_repo_access():
                return False

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

def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fast and reliable downloader for Hugging Face models and datasets."
    )
    parser.add_argument("repo_id_or_url", help="Repository ID or URL")
    parser.add_argument("-t", "--threads", type=int, default='auto',
                       help="Number of threads (default: auto)")
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
        resume=not args.no_resume
    )

    success = downloader.download()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
