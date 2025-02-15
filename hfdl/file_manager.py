import os
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger(__name__)

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
    """Manages file operations and tracking for downloads
    
    Handles:
    - File discovery and size checking
    - Small/big file categorization
    - Download progress tracking
    - Thread-safe operations
    """
    
    def __init__(self, api: HfApi, size_threshold_mb: float):
        self.api = api
        self.size_threshold_bytes = size_threshold_mb * 1024 * 1024
        self._files: Dict[str, FileInfo] = {}
        self._lock = threading.Lock()
        
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
            ValueError: If invalid repository type
        """
        try:
            # Get list of files from repository
            files = self.api.list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            
            small_files: List[FileInfo] = []
            big_files: List[FileInfo] = []
            
            # Get file sizes and categorize
            for file_path in files:
                try:
                    size = self.api.get_repo_file_metadata(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        path=file_path,
                        token=token
                    ).size
                    
                    file_info = FileInfo(
                        name=Path(file_path).name,
                        size=size,
                        path_in_repo=file_path,
                        local_path=Path(file_path)
                    )
                    
                    # Categorize based on size
                    if size <= self.size_threshold_bytes:
                        small_files.append(file_info)
                    else:
                        big_files.append(file_info)
                        
                    with self._lock:
                        self._files[file_path] = file_info
                        
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {file_path}: {e}")
                    continue
                    
            # Sort files by size (ascending for small, descending for big)
            small_files.sort(key=lambda x: x.size)
            big_files.sort(key=lambda x: x.size, reverse=True)
            
            logger.info(
                f"Discovered {len(small_files)} small files and "
                f"{len(big_files)} big files"
            )
            return small_files, big_files
            
        except RepositoryNotFoundError:
            logger.error(f"Repository not found: {repo_id}")
            raise
        except Exception as e:
            logger.error(f"Error discovering files: {e}")
            raise ValueError(f"Failed to discover files: {e}")
            
    def update_progress(self, file_path: str, bytes_downloaded: int):
        """Update download progress for a file
        
        Args:
            file_path: Path of file in repository
            bytes_downloaded: Number of bytes downloaded
        """
        with self._lock:
            if file_path in self._files:
                file_info = self._files[file_path]
                file_info.downloaded = bytes_downloaded
                file_info.completed = bytes_downloaded >= file_info.size
                
    def get_progress(self, file_path: str) -> Tuple[int, int]:
        """Get download progress for a file
        
        Args:
            file_path: Path of file in repository
            
        Returns:
            Tuple of (bytes_downloaded, total_bytes)
        """
        with self._lock:
            if file_path in self._files:
                file_info = self._files[file_path]
                return file_info.downloaded, file_info.size
            return 0, 0
            
    def is_completed(self, file_path: str) -> bool:
        """Check if file download is completed
        
        Args:
            file_path: Path of file in repository
        """
        with self._lock:
            if file_path in self._files:
                return self._files[file_path].completed
            return False
            
    def get_total_progress(self) -> Tuple[int, int]:
        """Get total download progress across all files
        
        Returns:
            Tuple of (total_bytes_downloaded, total_bytes)
        """
        with self._lock:
            total_downloaded = sum(f.downloaded for f in self._files.values())
            total_size = sum(f.size for f in self._files.values())
            return total_downloaded, total_size