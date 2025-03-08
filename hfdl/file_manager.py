import os
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    EntryNotFoundError,
    RevisionNotFoundError
)
from huggingface_hub.hf_api import RepoFile
from requests.exceptions import HTTPError
from .utils import sanitize_filename, get_os_compatible_path

logger = logging.getLogger(__name__)

class FileManagerError(Exception):
    """Base exception for file manager errors"""
    pass

class FileSizeError(FileManagerError):
    """Error when getting file size information"""
    pass

class FileTrackingError(FileManagerError):
    """Error when tracking file progress"""
    pass

@dataclass
class FileInfo:
    """Information about a file to be downloaded"""
    name: str
    size: int
    path_in_repo: str
    local_path: Path
    downloaded: int = 0
    completed: bool = False

class FileManager:
    """Manages file operations and tracking for downloads"""
    
    def __init__(self, api: HfApi, size_threshold_mb: float):
        try:
            self.api = api
            if size_threshold_mb <= 0:
                raise ValueError("Size threshold must be positive")
            self.size_threshold_bytes = size_threshold_mb * 1024 * 1024
            self._files: Dict[str, FileInfo] = {}
            self._lock = threading.Lock()
        except Exception as e:
            logger.error(f"Failed to initialize file manager: {e}")
            raise FileManagerError(f"File manager initialization failed: {e}")
        
    def discover_files(
        self,
        repo_id: str,
        repo_type: str,
        token: Optional[str] = None
    ) -> Tuple[List[FileInfo], List[FileInfo]]:
        """Discover and categorize files in repository
        
        Args:
            repo_id: Repository identifier
            repo_type: Type of repository (model, dataset, space)
            token: Optional authentication token
            
        Returns:
            Tuple of (small_files, big_files) lists
            
        Raises:
            RepositoryNotFoundError: If repository not found
            RevisionNotFoundError: If revision not found
            HTTPError: If network error occurs
            FileSizeError: If error getting file sizes
            FileManagerError: For other errors
        """
        try:
            # Get repository tree with file metadata in a single API call
            try:
                repo_files = self.api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                logger.error(f"Repository error: {e}")
                raise
            except HTTPError as e:
                logger.error(f"Network error listing repository tree: {e}")
                raise
            except Exception as e:
                logger.error(f"Error listing repository tree: {e}")
                raise FileManagerError(f"Failed to list repository tree: {e}")
            
            small_files: List[FileInfo] = []
            big_files: List[FileInfo] = []
            
            # Process file metadata and categorize
            for repo_file in repo_files:
                try:
                    # Skip directories
                    if repo_file.type == "directory":
                        continue
                        
                    # Create file info
                    try:
                        # Sanitize the path for OS compatibility
                        safe_path = get_os_compatible_path(repo_file.path)
                        file_info = FileInfo(
                            name=sanitize_filename(Path(repo_file.path).name),
                            size=repo_file.size,
                            path_in_repo=repo_file.path,
                            local_path=Path(safe_path)
                        )
                    except Exception as e:
                        logger.error(f"Error creating FileInfo for {repo_file.path}: {e}")
                        raise FileSizeError(f"Failed to process file info: {e}")
                    
                    # Categorize based on size
                    if repo_file.size <= self.size_threshold_bytes:
                        small_files.append(file_info)
                    else:
                        big_files.append(file_info)
                        
                    with self._lock:
                        # Store with original path as key for HF API reference
                        self._files[repo_file.path] = file_info
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing {repo_file.path}: {e}")
                    raise FileManagerError(f"Failed to process file: {e}")
                    
            # Sort files by size (ascending for small, descending for big)
            try:
                small_files.sort(key=lambda x: x.size)
                big_files.sort(key=lambda x: x.size, reverse=True)
            except Exception as e:
                logger.error(f"Error sorting files: {e}")
                raise FileManagerError(f"Failed to sort files: {e}")
            
            logger.info(
                f"Discovered {len(small_files)} small files and "
                f"{len(big_files)} big files"
            )
            return small_files, big_files
            
        except (RepositoryNotFoundError, RevisionNotFoundError, HTTPError, FileSizeError):
            raise
        except Exception as e:
            logger.error(f"Error discovering files: {e}")
            raise FileManagerError(f"Failed to discover files: {e}")
            
    def update_progress(self, file_path: str, bytes_downloaded: int):
        """Update download progress for a file
        
        Args:
            file_path: Path of file in repository
            bytes_downloaded: Number of bytes downloaded
            
        Raises:
            FileTrackingError: If error updating progress
        """
        try:
            with self._lock:
                if file_path in self._files:
                    file_info = self._files[file_path]
                    if bytes_downloaded < 0:
                        raise ValueError("Downloaded bytes cannot be negative")
                    file_info.downloaded = bytes_downloaded
                    file_info.completed = bytes_downloaded >= file_info.size
        except ValueError as e:
            logger.error(f"Invalid progress value: {e}")
            raise FileTrackingError(f"Invalid progress value: {e}")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
            raise FileTrackingError(f"Failed to update progress: {e}")
                
    def get_progress(self, file_path: str) -> Tuple[int, int]:
        """Get download progress for a file
        
        Args:
            file_path: Path of file in repository
            
        Returns:
            Tuple of (bytes_downloaded, total_bytes)
            
        Raises:
            FileTrackingError: If error getting progress
        """
        try:
            with self._lock:
                if file_path in self._files:
                    file_info = self._files[file_path]
                    return file_info.downloaded, file_info.size
                return 0, 0
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            raise FileTrackingError(f"Failed to get progress: {e}")
            
    def is_completed(self, file_path: str) -> bool:
        """Check if file download is completed
        
        Args:
            file_path: Path of file in repository
            
        Raises:
            FileTrackingError: If error checking completion
        """
        try:
            with self._lock:
                if file_path in self._files:
                    return self._files[file_path].completed
                return False
        except Exception as e:
            logger.error(f"Error checking completion: {e}")
            raise FileTrackingError(f"Failed to check completion: {e}")
            
    def get_total_progress(self) -> Tuple[int, int]:
        """Get total download progress across all files
        
        Returns:
            Tuple of (total_bytes_downloaded, total_bytes)
            
        Raises:
            FileTrackingError: If error calculating total progress
        """
        try:
            with self._lock:
                total_downloaded = sum(f.downloaded for f in self._files.values())
                total_size = sum(f.size for f in self._files.values())
                return total_downloaded, total_size
        except Exception as e:
            logger.error(f"Error calculating total progress: {e}")
            raise FileTrackingError(f"Failed to calculate total progress: {e}")