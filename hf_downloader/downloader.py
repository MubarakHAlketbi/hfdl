import argparse
import time
import requests
import os
import json
import re
import threading
import queue
import logging
import hashlib
import shutil
import multiprocessing
import random
import portalocker
import blake3
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_url, HfFolder
import signal
from tqdm import tqdm
from typing import Tuple, Optional, List, Set, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from datetime import datetime
from .config import DownloadConfig
import sys

logger = logging.getLogger(__name__)

class HFDownloadError(Exception):
    """Base exception for downloader errors"""
    pass

class HFDownloader:
    def __init__(
        self,
        model_id: str,
        download_dir: str = "downloads",
        num_threads: Union[int, str] = 'auto',
        repo_type: str = "model",
        chunk_size: int = 1024*1024,
        min_free_space: int = 5000,
        file_size_threshold: int = 200,
        min_speed_percentage: int = 5,
        speed_test_duration: int = 5,
        verify: bool = False,
        fix_broken: bool = False,
        force: bool = False
    ):
        self.model_id = model_id
        self.download_dir = Path(download_dir)
        self.repo_type = repo_type
        self.config = DownloadConfig(
            num_threads=num_threads if isinstance(num_threads, int) else 0,
            chunk_size=chunk_size,
            min_free_space_mb=min_free_space,
            file_size_threshold=file_size_threshold * 1024 * 1024,
            min_speed_percentage=min_speed_percentage,
            speed_test_duration=speed_test_duration,
            verify_downloads=verify,
            fix_broken=fix_broken,
            force_download=force
        )
        self.download_manager = None
        self.exit_event = threading.Event()

        signal.signal(signal.SIGINT, self._signal_handler)

    def download(self):
        """Main download method with proper error handling"""
        try:
            self.download_manager = DownloadManager(
                self.model_id,
                self.download_dir,
                self.repo_type,
                self.config
            )
            return self.download_manager.download()
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise HFDownloadError(str(e)) from e
        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources"""
        if self.download_manager:
            self.download_manager.exit_event.set()

    def _signal_handler(self, sig, frame):
        logger.info('Gracefully shutting down...')
        self.exit_event.set()
        if self.download_manager:
            self.download_manager.cleanup_resources()  # Add cleanup method in DownloadManager
        sys.exit(1)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5):
        self.failure_threshold = failure_threshold
        self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            logger.error("Circuit breaker triggered - too many consecutive failures")
            raise HFDownloadError("Too many download failures")

    def reset(self):
        self.failure_count = 0

class FileLocker:
    """Cross-platform file locking using portalocker."""
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.lock_path = self.path.parent / f".{self.path.name}.lock"
        self.lock_file = None

    def __enter__(self):
        """Acquire lock with retries."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.lock_file = open(self.lock_path, 'w')
                portalocker.lock(self.lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                return self
            except portalocker.LockException:
                if attempt < max_attempts - 1:
                    time.sleep(random.uniform(0.1, 0.3))  # Random backoff
                else:
                    raise
            except Exception as e:
                if self.lock_file:
                    self.lock_file.close()
                raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock and cleanup."""
        if self.lock_file:
            try:
                portalocker.unlock(self.lock_file)
                self.lock_file.close()
            finally:
                try:
                    self.lock_path.unlink(missing_ok=True)
                except Exception:
                    pass

class HybridHasher:
    """Hybrid hashing using BLAKE3 for speed and SHA256 for compatibility."""
    def __init__(self, size_threshold: int = 4 * 1024 * 1024 * 1024):  # 4GB threshold
        self.size_threshold = size_threshold

    def calculate_hash(self, file_path: Path, chunk_size: int = 8192) -> Tuple[str, str]:
        """Calculate both BLAKE3 and SHA256 hashes for large files, only SHA256 for small files."""
        size = file_path.stat().st_size
        sha256_hash = hashlib.sha256()
        blake3_hash = blake3.blake3() if size >= self.size_threshold else None

        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256_hash.update(chunk)
                if blake3_hash:
                    blake3_hash.update(chunk)

        return (
            blake3_hash.hexdigest() if blake3_hash else None,
            sha256_hash.hexdigest()
        )

logger = logging.getLogger(__name__)

class DownloadState:
    """Tracks and persists download state."""
    def __init__(self, repo_id: str, files: Dict[str, Dict[str, Any]]):
        self.repo_id = repo_id
        self.files = files or {}
        self.state_file: Optional[Path] = None

    @classmethod
    def create_new(cls, repo_id: str, file_list: List[Tuple[str, int, bool]]) -> 'DownloadState':
        """Create new state from file list (filename, size, is_lfs)."""
        files = {
            filename: {
                'size': size,
                'is_lfs': is_lfs,
                'status': 'pending',
                'downloaded': 0,
                'checksum': None,
                'last_modified': None
            } for filename, size, is_lfs in file_list
        }
        return cls(repo_id, files)

    def save(self) -> None:
        """Save current state to file."""
        if self.state_file:
            with self.state_file.open('w') as f:
                json.dump({
                    'repo_id': self.repo_id,
                    'files': self.files
                }, f)

    @classmethod
    def validate_schema(cls, data: Dict) -> bool:
        """Validate state file schema."""
        try:
            if not isinstance(data, dict):
                return False
            if 'repo_id' not in data or not isinstance(data['repo_id'], str):
                return False
            if 'files' not in data or not isinstance(data['files'], dict):
                return False
            
            for filename, file_info in data['files'].items():
                if not isinstance(filename, str):
                    return False
                if not isinstance(file_info, dict):
                    return False
                required_fields = {'size', 'is_lfs', 'status', 'downloaded'}
                if not all(field in file_info for field in required_fields):
                    return False
                if not isinstance(file_info['size'], int):
                    return False
                if not isinstance(file_info['is_lfs'], bool):
                    return False
                if not isinstance(file_info['status'], str):
                    return False
                if not isinstance(file_info['downloaded'], int):
                    return False
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, state_file: Path) -> Optional['DownloadState']:
        """Load state from file with schema validation."""
        try:
            with state_file.open() as f:
                data = json.load(f)
                
            if not cls.validate_schema(data):
                logger.error("Invalid state file schema")
                return None
                
            state = cls(data['repo_id'], data['files'])
            state.state_file = state_file
            return state
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in state file: {e}")
            return None
        except Exception as e:
            logger.error(f"Could not load state file: {e}")
            return None

    def update_file(self, filename: str, downloaded: int, status: str) -> None:
        """Update file progress."""
        if filename in self.files:
            self.files[filename].update({
                'downloaded': downloaded,
                'status': status,
                'last_modified': datetime.now().isoformat()
            })
            self.save()

    def set_state_file(self, path: Path) -> None:
        """Set state file path."""
        self.state_file = path

def verify_partial_file(file_path: Path, expected_size: int, expected_checksum: Optional[str] = None) -> Tuple[bool, int]:
    """Verify a partially downloaded file with hybrid hashing."""
    if not file_path.exists():
        return False, 0
    
    try:
        current_size = file_path.stat().st_size
        if current_size > expected_size:
            logger.warning(f"File {file_path} is larger than expected. Removing.")
            file_path.unlink()
            return False, 0
            
        if expected_checksum and current_size == expected_size:
            hasher = HybridHasher()
            blake3_hash, sha256_hash = hasher.calculate_hash(file_path)
            # Use SHA256 for compatibility with Hugging Face's API
            if sha256_hash != expected_checksum:
                logger.warning(f"Checksum mismatch for {file_path}. Removing.")
                file_path.unlink()
                return False, 0
            if blake3_hash:
                logger.debug(f"BLAKE3 hash calculated for large file {file_path}")
                
        return True, current_size
    except Exception as e:
        logger.error(f"Error verifying file {file_path}: {e}")
        return False, 0
    
def handle_errors(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {fn.__name__}: {str(e)}")
            # Only record failure if circuit_breaker exists
            if hasattr(args[0], 'circuit_breaker') and args[0].circuit_breaker is not None:
                args[0].circuit_breaker.record_failure()
            raise
    return wrapper

class SpeedTracker:
    """Tracks download speed using a rolling window."""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.bytes_window = deque()
        self.start_time = time.time()
        self.total_bytes = 0

    def update(self, bytes_downloaded: int):
        """Update speed tracking with new bytes."""
        current_time = time.time()
        self.bytes_window.append((current_time, bytes_downloaded))
        self.total_bytes += bytes_downloaded
        
        # Remove old entries outside the window
        while self.bytes_window and self.bytes_window[0][0] < current_time - self.window_size:
            _, old_bytes = self.bytes_window.popleft()
            self.total_bytes -= old_bytes

    def get_speed(self) -> float:
        """Get current speed in bytes per second."""
        if not self.bytes_window:
            return 0.0
        return self.total_bytes / min(time.time() - self.start_time, self.window_size)

    def reset(self):
        """Reset speed tracking."""
        self.bytes_window.clear()
        self.total_bytes = 0
        self.start_time = time.time()

class RateLimiter:
    """Limits download rate per thread."""
    def __init__(self, max_speed: float):  # max_speed in bytes/sec
        self.max_speed = max_speed
        self.downloaded = 0
        self.start_time = time.time()

    def limit(self, chunk_size: int):
        """Apply rate limiting."""
        self.downloaded += chunk_size
        elapsed = time.time() - self.start_time
        expected_time = self.downloaded / self.max_speed
        sleep_duration = expected_time - elapsed
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    def update_limit(self, new_max_speed: float):
        """Update the rate limit."""
        self.max_speed = new_max_speed
        # Reset counters to avoid sudden speed changes
        self.downloaded = 0
        self.start_time = time.time()

class FileClassifier:
    """Classifies files based on size for optimized downloading."""
    def __init__(self, file_size_threshold: int, repo_id: str, repo_type: str):
        self.file_size_threshold = file_size_threshold
        self.repo_id = repo_id
        self.repo_type = repo_type

    def _get_fallback_size(self, filename: str) -> int:
        """Fallback size retrieval when API doesn't provide it"""
        try:
            url = hf_hub_url(repo_id=self.repo_id, filename=filename, repo_type=self.repo_type)
            response = requests.head(url, timeout=10)
            return int(response.headers.get('Content-Length', 0))
        except Exception as e:
            logger.warning(f"Couldn't get size for {filename}: {e}")
            return 0

    def classify_files(self, files: List[Tuple[str, int, bool]]) -> Tuple[List[Tuple[str, int, bool]], List[Tuple[str, int, bool]]]:
        """Classify files into small and big based on size threshold."""
        small_files = []
        big_files = []
        
        for file_info in files:
            filename, size, is_lfs = file_info
            if size is None:
                size = self._get_fallback_size(filename)
            
            if size < self.file_size_threshold:
                small_files.append((filename, size, is_lfs))
            else:
                big_files.append((filename, size, is_lfs))
        
        return small_files, big_files
class SpeedTestManager:
    """Manages speed testing for downloads."""
    def __init__(self, duration: int, speed_tracker: SpeedTracker):
        """Initialize speed test manager."""
        self.duration = duration
        self.speed_tracker = speed_tracker
        self.start_time = 0
        self.test_complete = False

    def start_test(self):
        """Start speed test."""
        self.start_time = time.time()
        self.test_complete = False
        if hasattr(self.speed_tracker, 'reset'):
            self.speed_tracker.reset()

    def update(self, bytes_downloaded: int) -> bool:
        """Update speed test with new bytes. Returns True if test is complete."""
        if self.test_complete:
            return True

        self.speed_tracker.update(bytes_downloaded)
        elapsed = time.time() - self.start_time

        if elapsed >= self.duration:
            self.test_complete = True
            return True
        return False

    def get_average_speed(self) -> float:
        """Get average speed during test period in bytes per second."""
        return self.speed_tracker.get_average_speed()

class ThreadOptimizer:
    """Optimizes thread count based on speed requirements."""
    def __init__(self, min_speed_percentage: float, max_threads: int):
        self.min_speed_percentage = min_speed_percentage
        self.max_threads = max_threads

    def calculate_optimal_threads(self, total_speed: float) -> int:
        """Calculate optimal number of threads based on speed requirements."""
        if total_speed <= 0:
            return 1

        # Calculate minimum speed per thread as percentage of total speed
        min_speed_per_thread = (total_speed * self.min_speed_percentage) / 100.0
        
        # Ensure minimum speed is not too low (at least 1 MB/s)
        min_speed_per_thread = max(min_speed_per_thread, 1024 * 1024)
        
        # Calculate how many threads we can support while maintaining min_speed_per_thread
        optimal_threads = int(total_speed / min_speed_per_thread)
        
        # Ensure we stay within bounds
        optimal_threads = max(1, min(optimal_threads, self.max_threads))
        
        logger.info(
            f"Speed-based thread calculation:"
            f"\n- Total speed: {total_speed / (1024*1024):.2f} MB/s"
            f"\n- Target speed percentage: {self.min_speed_percentage:.1f}%"
            f"\n- Min speed per thread: {min_speed_per_thread / (1024*1024):.2f} MB/s"
            f"\n- Optimal threads: {optimal_threads}"
        )
        
        return optimal_threads

class DownloadManager:
    @staticmethod
    def _normalize_repo_id(repo_id_or_url: str) -> str:
        """Normalize repository ID from URL or direct input with validation."""
        if not repo_id_or_url:
            raise ValueError("Repository ID cannot be empty")
            
        # Remove URL prefix if present
        repo_id = repo_id_or_url.replace("https://huggingface.co/", "")
        repo_id = repo_id.rstrip("/")
        
        # Validate repository ID format
        if not re.match(r'^[\w.-]+/[\w.-]+$', repo_id):
            raise ValueError(
                f"Invalid repository ID format: {repo_id}\n"
                "Expected format: username/repository-name\n"
                "Allowed characters: letters, numbers, hyphens, underscores, and dots"
            )
            
        return repo_id

    @staticmethod
    def _get_auth_token() -> Optional[str]:
        token = HfFolder.get_token()
        if token:
            logger.info("Using authentication token")
            # Enhanced token masking
            if len(token) > 8:
                masked_token = f"{token[:4]}...{token[-4:]}"
                logger.debug(f"Using token: {masked_token}")
            else:
                logger.debug("Using token: ********")
        return token

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_resources()
        
    def cleanup_resources(self):
        """Proper cleanup of network connections and file locks"""
        try:
            # Signal all threads to stop
            self.exit_event.set()
            
            # Wait briefly for threads to notice the exit signal
            cleanup_timeout = 5  # seconds
            cleanup_start = time.time()
            
            # Check if downloads have stopped
            while time.time() - cleanup_start < cleanup_timeout:
                if not any(t.get_speed() > 0 for t in getattr(self, 'speed_trackers', {}).values()):
                    break
                time.sleep(0.1)
            
            # Force shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False, cancel_futures=True)
            
            # Clean up any remaining locks
            for lock in getattr(self, 'file_locks', {}).values():
                try:
                    if hasattr(lock, 'release'):
                        lock.release()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _signal_handler(self, sig, frame):
        """Handle interrupt signal."""
        logger.info('Received interrupt signal (Ctrl+C). Stopping downloads...')
        self.exit_event.set()
        
        # Log active downloads
        active_files = [f for f, t in self.speed_trackers.items() if t.get_speed() > 0]
        if active_files:
            logger.info("Active downloads being stopped:")
            for file in active_files:
                if file in self.download_state.files:
                    progress = self.download_state.files[file]['downloaded']
                    total = self.download_state.files[file]['size']
                    logger.info(f"- {file}: {progress}/{total} bytes ({progress/total*100:.1f}%)")
        
        logger.info("Download state will be saved. Resume with the same command to continue.")
        
        # Give time for cleanup operations
        cleanup_timeout = 5  # seconds
        cleanup_start = time.time()
        
        while time.time() - cleanup_start < cleanup_timeout:
            # Check if all downloads have stopped
            if not any(t.get_speed() > 0 for t in self.speed_trackers.values()):
                logger.info("All downloads stopped successfully")
                break
            time.sleep(0.1)
        
        if any(t.get_speed() > 0 for t in self.speed_trackers.values()):
            logger.warning("Some downloads are still active. State will be saved but some files may be incomplete.")

    def __init__(self, repo_id: str, output_dir: Path, repo_type: str, config: DownloadConfig):
        self.repo_id = self._normalize_repo_id(repo_id)
        self.output_dir = output_dir / self.repo_id.split('/')[-1]
        self.repo_type = repo_type
        self.config = config
        self.exit_event = threading.Event()
        self.completed_files = set()
        self.download_queue = queue.Queue()
        self.file_locks = {}
        self.token = None
        self.speed_tracker = SpeedTracker()
        self.download_state = None
        self.total_speed_tracker = SpeedTracker()
        self.speed_test_manager = SpeedTestManager(
            duration=config.speed_test_duration,
            speed_tracker=SpeedTracker()
        )

        # Initialize components
        self.file_classifier = FileClassifier(
            file_size_threshold=self.config.file_size_threshold,
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
        self.circuit_breaker = CircuitBreaker()
        signal.signal(signal.SIGINT, self._signal_handler)

    def _ensure_directory(self, path: Path) -> bool:
        """Ensure directory exists with proper locking and parent directory checks."""
        dir_path = str(path)
        if dir_path not in self.dir_locks:
            self.dir_locks[dir_path] = threading.Lock()
            
        with self.dir_locks[dir_path]:
            try:
                if not path.exists():
                    # Check and create parent directories first
                    if not path.parent.exists():
                        if not self._ensure_directory(path.parent):
                            return False
                    path.mkdir(mode=0o755)
                return True
            except Exception as e:
                logger.error(f"Failed to create directory {path}: {e}")
                return False

    def _download_small_files(self, files: List[Tuple[str, int, bool]]) -> bool:
        """Download small files using all available threads."""
        logger.info("Starting download of small files using all available threads...")
        
        # Clear and prepare download queue
        while not self.download_queue.empty():
            self.download_queue.get()
        
        # Add all small files to queue
        for file_info in files:
            if file_info[0] not in self.completed_files:
                self.download_queue.put((file_info[0], file_info[2]))  # filename and is_lfs
        
        if self.download_queue.empty():
            logger.info("No small files to download")
            return True
        
        # Use all available threads
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            workers = [executor.submit(self._worker) for _ in range(self.config.num_threads)]
            try:
                self.download_queue.join()
                return True
            except Exception as e:
                logger.error(f"Error downloading small files: {e}")
                return False

    def _test_download_speed(self, file_info: Tuple[str, int, bool]) -> float:
        """Test download speed using a single file."""
        filename, size, is_lfs = file_info
        logger.info(f"Testing download speed with file: {filename}")
        
        self.speed_test_manager.start_test()
        
        # Clear and prepare download queue
        while not self.download_queue.empty():
            self.download_queue.get()
        
        # Use single thread for speed test
        self.download_queue.put((filename, is_lfs))
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = executor.submit(self._worker)
            try:
                self.download_queue.join()
                avg_speed = self.speed_test_manager.get_average_speed()
                logger.info(f"Speed test complete. Average speed: {avg_speed / (1024*1024):.2f} MB/s")
                return avg_speed
            except Exception as e:
                logger.error(f"Error during speed test: {e}")
                return 0.0

    def _download_big_files(self, files: List[Tuple[str, int, bool]], thread_count: int) -> bool:
        """Download big files with optimized thread count."""
        logger.info(f"Starting download of big files using {thread_count} threads...")
        
        # Clear and prepare download queue
        while not self.download_queue.empty():
            self.download_queue.get()
        
        # Add all big files to queue
        for file_info in files:
            if file_info[0] not in self.completed_files:
                self.download_queue.put((file_info[0], file_info[2]))  # filename and is_lfs
        
        if self.download_queue.empty():
            logger.info("No big files to download")
            return True
        
        # Use optimized thread count
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            workers = [executor.submit(self._worker) for _ in range(thread_count)]
            try:
                self.download_queue.join()
                return True
            except Exception as e:
                logger.error(f"Error downloading big files: {e}")
                return False

    def _initialize_state(self, files: List[Tuple[str, int, bool]]) -> None:
        """Initialize or load download state."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            state_file = self.output_dir / '.download_state'
            
            # Try to load existing state file
            if state_file.exists() and not self.config.force_download:
                loaded_state = DownloadState.load(state_file)
                if loaded_state and loaded_state.repo_id == self.repo_id:
                    self.download_state = loaded_state
                    logger.info("Resumed previous download state")
                    return

            # Create new state
            self.download_state = DownloadState.create_new(self.repo_id, files)
            self.download_state.set_state_file(state_file)
            
            # Check for existing files from previous downloads
            if self.output_dir.exists() and not self.config.force_download:
                logger.info("Found existing files from previous download, verifying...")
                self._verify_existing_files()
            
            # Save the state
            self.download_state.save()
            
        except Exception as e:
            logger.error(f"Error initializing state: {e}")
            raise

    def _verify_existing_files(self) -> None:
        """Verify all existing downloaded files."""
        if not self.download_state:
            return

        for filename, info in self.download_state.files.items():
            file_path = self.output_dir / filename
            valid, size = verify_partial_file(file_path, info['size'], info.get('checksum'))
            
            if not valid and self.config.fix_broken:
                logger.info(f"Removing invalid file: {filename}")
                info['status'] = 'pending'
                info['downloaded'] = 0
            elif valid and size == info['size']:
                logger.info(f"Verified complete file: {filename}")
                info['status'] = 'completed'
                self.completed_files.add(filename)
            elif valid:
                logger.info(f"Found partial download: {filename} ({size}/{info['size']} bytes)")
                info['status'] = 'pending'
                info['downloaded'] = size

        self.download_state.save()

    def _update_rate_limiters(self):
        """Update rate limiters based on current speed with smoother adjustments."""
        total_speed = self.total_speed_tracker.get_speed()
        active_limiters = len(self.rate_limiters)
        
        if active_limiters > 0:
            # Constants for bandwidth management
            MIN_SPEED_PER_THREAD = 3 * 1024 * 1024  # 3 MB/s minimum per thread
            MAX_SPEED_PER_THREAD = 50 * 1024 * 1024  # 50 MB/s maximum per thread
            ALPHA = 0.2  # Exponential moving average factor
            
            # Take 95% of total speed for stability
            available_speed = total_speed * 0.95
            
            # Calculate optimal number of threads based on speed
            optimal_threads = max(1, int(available_speed / MIN_SPEED_PER_THREAD))
            
            # If we have more active limiters than optimal, reduce speed instead of threads
            if active_limiters > optimal_threads:
                target_speed = available_speed / active_limiters
            else:
                target_speed = available_speed / optimal_threads
            
            # Ensure minimum speed per thread
            target_speed = max(MIN_SPEED_PER_THREAD, min(MAX_SPEED_PER_THREAD, target_speed))
            
            # Update each rate limiter with smoothing
            for limiter in self.rate_limiters.values():
                current_limit = limiter.max_speed
                # Exponential moving average for smoother transitions
                new_limit = (ALPHA * target_speed) + ((1 - ALPHA) * current_limit)
                # Ensure we stay within bounds
                new_limit = max(MIN_SPEED_PER_THREAD, min(MAX_SPEED_PER_THREAD, new_limit))
                limiter.update_limit(new_limit)
            
            # Calculate actual allocation for logging
            total_allocated = sum(limiter.max_speed for limiter in self.rate_limiters.values())
            avg_thread_speed = total_allocated / active_limiters
            
            # Only log significant bandwidth changes (>20% change)
            if not hasattr(self, '_last_logged_speed') or \
               abs(total_speed - getattr(self, '_last_logged_speed', 0)) / max(total_speed, 1) > 0.2:
                self._last_logged_speed = total_speed
                logger.info(
                    f"Bandwidth adjusted:"
                    f"\n- Total speed: {total_speed / (1024*1024):.2f} MB/s"
                    f"\n- Available speed (95%): {available_speed / (1024*1024):.2f} MB/s"
                    f"\n- Active threads: {active_limiters}"
                    f"\n- Optimal threads: {optimal_threads}"
                    f"\n- Per-thread speed: {avg_thread_speed / (1024*1024):.2f} MB/s"
                )

    def _process_chunk(self, chunk: bytes, filename: str):
        """Process a downloaded chunk, updating speed tracking."""
        chunk_size = len(chunk)
        
        # Update speed trackers
        if filename not in self.speed_trackers:
            self.speed_trackers[filename] = SpeedTracker()
        self.speed_trackers[filename].update(chunk_size)
        self.total_speed_tracker.update(chunk_size)
        
        # Apply rate limiting
        if filename not in self.rate_limiters:
            # Initialize with current speed
            current_speed = self.speed_trackers[filename].get_speed()
            self.rate_limiters[filename] = RateLimiter(current_speed)
        
        self.rate_limiters[filename].limit(chunk_size)

    def _worker(self):
        """Worker function for download thread."""
        while not self.exit_event.is_set():
            try:
                filename, is_lfs = self.download_queue.get(timeout=1)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")
                break

            local_path = self.output_dir / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Use cross-platform file locking
            with FileLocker(local_path):
                # Pre-allocate space if we know the file size
                if self.download_state and filename in self.download_state.files:
                    total_size = self.download_state.files[filename]['size']
                    if not local_path.exists() and total_size > 0:
                        if not self._preallocate_file(local_path, total_size):
                            logger.error(f"Failed to pre-allocate space for {filename}")
                            return False

                success = (self._download_lfs_file(filename, local_path) if is_lfs
                          else self._download_regular_file(filename, local_path))

                if success:
                    with threading.Lock():
                        self.completed_files.add(filename)

            self.download_queue.task_done()

    def _download_file_with_speed_control(self, response: requests.Response, 
                                        local_path: Path, filename: str,
                                        total_size: int, existing_size: int = 0) -> bool:
        """Download file with speed monitoring and control."""
        try:
            with local_path.open('ab') as f:
                with tqdm(total=total_size, initial=existing_size, unit='B',
                         unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
                    try:
                        for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                            if self.exit_event.is_set():
                                return False
                            if chunk:
                                self._process_chunk(chunk, filename)
                                f.write(chunk)
                                chunk_size = len(chunk)
                                pbar.update(chunk_size)
                                
                                # Update download state
                                if self.download_state:
                                    current_size = existing_size + pbar.n
                                    self.download_state.update_file(
                                        filename, current_size, 'in_progress'
                                    )
                                
                                # Periodically update rate limiters
                                if time.time() % self.config.speed_check_interval < 1:
                                    self._update_rate_limiters()
                    except Exception as e:
                        pbar.close()
                        raise
            
            # Update final state
            if self.download_state:
                self.download_state.update_file(
                    filename, total_size, 'completed'
                )
            return True
        except Exception as e:
            logger.error(f"Error downloading chunk for {filename}: {e}")
            if self.download_state:
                self.download_state.update_file(
                    filename, existing_size + pbar.n, 'failed'
                )
            return False

    def _download_with_retry(self, url: str, headers: dict, timeout: Tuple[int, int],
                           stream: bool = False) -> requests.Response:
        """Download with exponential backoff retry and detailed logging."""
        last_error = None
        total_wait_time = 0
        
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries}")
                
                response = requests.get(url, stream=stream, headers=headers,
                                     timeout=timeout, verify=True)
                response.raise_for_status()
                
                if attempt > 0:
                    logger.info(f"Successfully recovered after {attempt + 1} attempts "
                              f"(total wait time: {total_wait_time}s)")
                return response
                
            except requests.exceptions.ConnectTimeout as e:
                last_error = f"Connection timeout: {e}"
                logger.warning(f"Connection timeout on attempt {attempt + 1}/{self.config.max_retries}")
            except requests.exceptions.ReadTimeout as e:
                last_error = f"Read timeout: {e}"
                logger.warning(f"Read timeout on attempt {attempt + 1}/{self.config.max_retries}")
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.config.max_retries}")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error {e.response.status_code}: {e}"
                logger.warning(f"HTTP error {e.response.status_code} on attempt {attempt + 1}/{self.config.max_retries}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning(f"Unexpected error on attempt {attempt + 1}/{self.config.max_retries}: {e}")
            
            if attempt < self.config.max_retries - 1:
                wait_time = min(2 ** attempt, 60)  # Cap wait time at 60 seconds
                total_wait_time += wait_time
                logger.warning(f"Retrying in {wait_time} seconds... "
                             f"({attempt + 1}/{self.config.max_retries} attempts)")
                time.sleep(wait_time)
            else:
                logger.error(f"All retry attempts exhausted. Last error: {last_error}")
                logger.error(f"Total time spent retrying: {total_wait_time} seconds")
                raise requests.exceptions.RequestException(
                    f"Failed after {self.config.max_retries} attempts. {last_error}"
                )

    def _check_disk_space(self, required_mb: int) -> bool:
        """Check if there's enough disk space available with 5% safety buffer."""
        try:
            total, used, free = shutil.disk_usage(self.output_dir.parent)
            free_mb = free // (1024 * 1024)
            
            # Add 5% buffer to required space
            required_with_buffer = int(required_mb * 1.05)
            
            if free_mb < required_with_buffer:
                logger.error(
                    f"Insufficient disk space:"
                    f"\n- Required: {required_mb} MB"
                    f"\n- Required with 5% buffer: {required_with_buffer} MB"
                    f"\n- Available: {free_mb} MB"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return False

    def _preallocate_file(self, file_path: Path, size: int) -> bool:
        """Pre-allocate file space to prevent fragmentation and ENOSPC errors."""
        try:
            with file_path.open('wb') as f:
                if os.name == 'posix':
                    # Use posix_fallocate on Unix systems
                    os.posix_fallocate(f.fileno(), 0, size)
                else:
                    # Use seek/truncate on Windows
                    f.seek(size - 1)
                    f.write(b'\0')
            return True
        except Exception as e:
            logger.error(f"Failed to pre-allocate space for {file_path}: {e}")
            return False

    def _verify_file_integrity(self, file_path: Path, expected_size: int, filename: str) -> bool:
        """Verify downloaded file integrity using hybrid hashing."""
        try:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                logger.error(f"Size mismatch for {filename}: expected {expected_size}, got {actual_size}")
                return False

            # Calculate checksum for regular files
            if not filename.endswith('.gitattributes'):  # Skip LFS pointer files
                logger.info(f"Calculating checksums for {filename}...")
                blake3_hash, sha256_hash = self.hasher.calculate_hash(file_path)
                
                # Store SHA256 checksum in state
                if sha256_hash and filename in self.download_state.files:
                    self.download_state.files[filename]['checksum'] = sha256_hash
                    self.download_state.save()
                
                # Log if BLAKE3 was used (for large files)
                if blake3_hash:
                    logger.info(f"Used BLAKE3 for fast hashing of large file {filename}")
                
                # Refresh token if needed
                self._refresh_token_if_needed()
                
                # Get expected checksum from server
                url = hf_hub_url(repo_id=self.repo_id, filename=filename, repo_type=self.repo_type)
                headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
                
                try:
                    response = requests.head(url, headers=headers,
                                          timeout=(self.config.connect_timeout, self.config.read_timeout),
                                          verify=True)
                    
                    if 'X-Content-Hash' in response.headers:
                        expected_checksum = response.headers['X-Content-Hash']
                        # Always use SHA256 for API compatibility
                        if sha256_hash != expected_checksum:
                            logger.error(f"Checksum mismatch for {filename}")
                            if filename in self.download_state.files:
                                self.download_state.files[filename]['status'] = 'checksum_failed'
                                self.download_state.save()
                            return False
                        logger.info(f"Checksum verified for {filename}")
                        # Update state with verified checksum
                        if filename in self.download_state.files:
                            self.download_state.files[filename]['checksum'] = expected_checksum
                            self.download_state.files[filename]['status'] = 'verified'
                            self.download_state.save()
                    else:
                        logger.warning(f"No checksum available from server for {filename}, using size verification only")
                        if filename in self.download_state.files:
                            self.download_state.files[filename]['status'] = 'size_verified'
                            self.download_state.save()
                except Exception as e:
                    logger.warning(f"Could not verify checksum for {filename}: {e}")
                    if filename in self.download_state.files:
                        self.download_state.files[filename]['status'] = 'verification_failed'
                        self.download_state.save()
            
            return True
        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False

    def _parse_lfs_pointer(self, pointer_content: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse Git LFS pointer file with improved validation."""
        try:
            if not pointer_content or not isinstance(pointer_content, str):
                logger.error("Invalid LFS pointer content")
                return None, None
                
            match = re.match(
                r"version https://git-lfs\.github\.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize (\d+)",
                pointer_content.strip(),
            )
            if match:
                try:
                    size = int(match.group(2))
                    if size < 0:
                        logger.error("Invalid negative size in LFS pointer")
                        return None, None
                    return match.group(1), size
                except ValueError:
                    logger.error("Invalid size format in LFS pointer")
                    return None, None
            else:
                logger.error("Invalid LFS pointer format")
                return None, None
        except Exception as e:
            logger.error(f"Error parsing LFS pointer: {e}")
            return None, None

    def _download_lfs_file(self, filename: str, local_path: Path) -> bool:
        """Download LFS file with resume capability."""
        try:
            # Refresh token if needed
            self._refresh_token_if_needed()
            
            # Download and parse pointer file
            pointer_url = hf_hub_url(repo_id=self.repo_id, filename=filename, repo_type=self.repo_type)
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            
            response = self._download_with_retry(pointer_url, headers=headers,
                                               timeout=(self.config.connect_timeout, self.config.read_timeout))
            oid, size = self._parse_lfs_pointer(response.text)
            
            if not oid:
                logger.error(f"Invalid LFS pointer for {filename}")
                return False

            # Check disk space
            if not self._check_disk_space(size // (1024 * 1024) + self.config.min_free_space_mb):
                logger.error(f"Insufficient disk space for {filename}")
                return False

            # Get download URL from LFS batch API
            batch_url = f"https://huggingface.co/{self.repo_id}/info/lfs/objects/batch"
            batch_data = {
                "operation": "download",
                "transfers": ["basic"],
                "objects": [{"oid": oid, "size": size}],
            }
            
            response = self._download_with_retry(batch_url, headers=headers,
                                               timeout=(self.config.connect_timeout, self.config.read_timeout))
            download_info = response.json()["objects"][0]["actions"]["download"]
            
            # Prepare for download with resume support
            download_headers = download_info.get("header", {})
            if local_path.exists():
                existing_size = local_path.stat().st_size
                download_headers["Range"] = f"bytes={existing_size}-"
            else:
                existing_size = 0

            # Download file with speed control
            with self._download_with_retry(download_info["href"], headers=download_headers,
                                         timeout=(self.config.connect_timeout, self.config.read_timeout),
                                         stream=True) as r:
                
                success = self._download_file_with_speed_control(
                    r, local_path, filename, size, existing_size
                )

                if not success:
                    return False

            # Verify downloaded file
            if not self._verify_file_integrity(local_path, size):
                logger.error(f"File integrity check failed for {filename}")
                local_path.unlink(missing_ok=True)
                return False

            return True

        except Exception as e:
            logger.error(f"Error downloading LFS file {filename}: {e}")
            return False

    def _download_regular_file(self, filename: str, local_path: Path) -> bool:
        """Download regular (non-LFS) file with resume capability."""
        try:
            # Refresh token if needed
            self._refresh_token_if_needed()
            
            url = hf_hub_url(repo_id=self.repo_id, filename=filename, repo_type=self.repo_type)
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

            # Check if file exists and verify size
            existing_size = 0
            if local_path.exists():
                try:
                    existing_size = local_path.stat().st_size
                    # Get file size without downloading
                    head_response = requests.head(url, headers=headers,
                                               timeout=(self.config.connect_timeout, self.config.read_timeout),
                                               verify=True)
                    total_size = int(head_response.headers.get('Content-Length', 0))
                    
                    # If file is complete, skip download
                    if existing_size == total_size:
                        logger.info(f"File {filename} is already complete")
                        return True
                    
                    # If file exists but is not complete, try resume
                    if existing_size < total_size:
                        # Test if server supports range requests
                        test_headers = headers.copy()
                        test_headers["Range"] = "bytes=0-0"
                        try:
                            test_response = requests.head(url, headers=test_headers,
                                                      timeout=(self.config.connect_timeout, self.config.read_timeout),
                                                      verify=True)
                            
                            if test_response.status_code == 206:
                                # Server supports range requests
                                headers["Range"] = f"bytes={existing_size}-"
                                logger.info(f"Resuming download of {filename} from byte {existing_size}")
                            else:
                                # Server doesn't support range requests, start over
                                logger.warning(f"Server doesn't support partial downloads for {filename}. Starting fresh download.")
                                local_path.unlink()
                                existing_size = 0
                        except Exception as e:
                            logger.warning(f"Error testing range support for {filename}: {e}. Starting fresh download.")
                            local_path.unlink()
                            existing_size = 0
                    else:
                        # File is larger than expected, remove and start over
                        logger.warning(f"File {filename} is larger than expected. Removing.")
                        local_path.unlink()
                        existing_size = 0
                except Exception as e:
                    logger.warning(f"Error checking file size for {filename}: {e}. Starting fresh download.")
                    local_path.unlink(missing_ok=True)
                    existing_size = 0

            with self._download_with_retry(url, headers=headers,
                                         timeout=(self.config.connect_timeout, self.config.read_timeout),
                                         stream=True) as r:
                
                total_size = int(r.headers.get('Content-Length', 0)) + existing_size
                
                # Check disk space
                if not self._check_disk_space(total_size // (1024 * 1024) + self.config.min_free_space_mb):
                    logger.error(f"Insufficient disk space for {filename}")
                    return False

                return self._download_file_with_speed_control(
                    r, local_path, filename, total_size, existing_size
                )

        except Exception as e:
            logger.error(f"Error downloading file {filename}: {e}")
            return False

    def _try_repo_access(self, token: Optional[str] = None) -> Optional[Any]:
        """Try to access repository with or without token."""
        try:
            api = HfApi()
            return api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type, token=token)
        except Exception as e:
            if "401" in str(e):  # Authentication required
                if not token:
                    logger.info("Repository requires authentication, trying with token...")
                    token = self._get_auth_token()
                    if token:
                        return self._try_repo_access(token)
                    else:
                        logger.error("This repository requires authentication. Please login using 'huggingface-cli login'")
                        return None
                else:
                    logger.error("Authentication failed. Please check your Hugging Face token.")
                    return None
            elif "404" in str(e):
                logger.error(f"Repository '{self.repo_id}' not found")
                return None
            else:
                logger.error(f"Error accessing repository: {e}")
                return None

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        optimal_threads = self.config.calculate_optimal_threads()
        available_threads = multiprocessing.cpu_count()
        
        logger.info("Thread configuration:")
        if available_threads <= 2:
            logger.info("- Download threads: 1")
            logger.info("- Reserved for Ctrl+C: 1")
            logger.info("- Reserved free: 0")
            return 1
        else:
            logger.info(f"- Download threads: {optimal_threads}")
            logger.info("- Reserved for Ctrl+C: 1")
            logger.info("- Reserved free: 1")
            return optimal_threads

    def _refresh_token_if_needed(self) -> None:
        """Refresh token if it's expired."""
        if not hasattr(self, 'token_last_refresh'):
            self.token_last_refresh = time.time()
            return

        current_time = time.time()
        if (current_time - self.token_last_refresh) >= self.config.token_refresh_interval:
            logger.info("Refreshing authentication token...")
            new_token = self._get_auth_token()
            if new_token != self.token:
                logger.info("Token updated")
                self.token = new_token
            self.token_last_refresh = current_time

    def download(self) -> bool:
        """Main download workflow with enhanced error handling"""
        try:
            if not self._initialize():
                return False

            files = self._get_file_list()
            if not files:
                logger.error("No files found for download")
                return False

            self._classify_files(files)
            return self._download_files()
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise

    def _initialize(self) -> bool:
        """Initialize the download environment"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check disk space upfront
            if not self._check_disk_space(1024):  # Check minimum 1GB
                return False

            # Get authentication token
            self.token = HfFolder.get_token()
            
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Fast and reliable downloader for Hugging Face models and datasets."
    )
    parser.add_argument("repo_id_or_url", nargs='?', help="Repository ID or URL")
    parser.add_argument("-t", "--threads", type=int, 
                       default=DownloadConfig.calculate_optimal_threads(),
                       help="Number of threads (default: auto-detected based on system)")
    parser.add_argument("-d", "--directory", default="downloads", help="Download directory")
    parser.add_argument("-r", "--repo_type", default="model",
                       choices=["model", "dataset", "space"], help="Repository type")
    parser.add_argument("--chunk-size", type=int, default=1024*1024,
                       help="Download chunk size in bytes")
    parser.add_argument("--min-free-space", type=int, default=1000,
                       help="Minimum required free space in MB")
    parser.add_argument("--verify", action="store_true",
                       help="Verify existing downloads")
    parser.add_argument("--fix-broken", action="store_true",
                       help="Remove and redownload corrupted files")
    parser.add_argument("--force", action="store_true",
                       help="Force fresh download, ignore existing files")
    parser.add_argument("--file-size-threshold", type=int, default=200,
                       help="Size threshold for big files in MB (default: 200)")
    parser.add_argument("--min-speed-percentage", type=float, default=5.0,
                        help="Target minimum speed per thread as percentage of average speed (1-100, default: 5)")
    parser.add_argument("--speed-test-duration", type=int, default=5,
                       help="Duration of speed test in seconds (default: 5)")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Maximum download retry attempts")
    parser.add_argument("--connect-timeout", type=int, default=10,
                       help="Connection timeout in seconds")
    parser.add_argument("--read-timeout", type=int, default=30,
                       help="Read timeout in seconds")
    args = parser.parse_args()

    if not args.repo_id_or_url:
        args.repo_id_or_url = input("Enter the Hugging Face repository ID or URL: ")

    config = DownloadConfig(
        num_threads=args.threads,
        chunk_size=args.chunk_size,
        min_free_space_mb=args.min_free_space,
        verify_downloads=args.verify,
        fix_broken=args.fix_broken,
        force_download=args.force,
        file_size_threshold=args.file_size_threshold * 1024 * 1024,  # Convert MB to bytes
        min_speed_percentage=args.min_speed_percentage,
        speed_test_duration=args.speed_test_duration,
        max_retries=args.max_retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout
    )

    manager = DownloadManager(
        repo_id=args.repo_id_or_url,
        output_dir=args.directory,
        repo_type=args.repo_type,
        config=config
    )

    success = manager.download()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
