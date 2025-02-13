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
from concurrent.futures import ThreadPoolExecutor
import argparse
from huggingface_hub import HfApi, hf_hub_url, scan_cache_dir
import signal
from tqdm import tqdm
from typing import Tuple, Optional, List, Set, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    def load(cls, state_file: Path) -> Optional['DownloadState']:
        """Load state from file."""
        try:
            with state_file.open() as f:
                data = json.load(f)
                state = cls(data['repo_id'], data['files'])
                state.state_file = state_file
                return state
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")
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

def calculate_file_checksum(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def verify_partial_file(file_path: Path, expected_size: int, expected_checksum: Optional[str] = None) -> Tuple[bool, int]:
    """Verify a partially downloaded file."""
    if not file_path.exists():
        return False, 0
    
    try:
        current_size = file_path.stat().st_size
        if current_size > expected_size:
            logger.warning(f"File {file_path} is larger than expected. Removing.")
            file_path.unlink()
            return False, 0
            
        if expected_checksum and current_size == expected_size:
            actual_checksum = calculate_file_checksum(file_path)
            if actual_checksum != expected_checksum:
                logger.warning(f"Checksum mismatch for {file_path}. Removing.")
                file_path.unlink()
                return False, 0
                
        return True, current_size
    except Exception as e:
        logger.error(f"Error verifying file {file_path}: {e}")
        return False, 0

class SpeedTracker:
    """Tracks download speed with rolling window."""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size  # seconds
        self.bytes_window = deque()
        self.start_time = time.time()
        self.total_bytes = 0

    def update(self, bytes_downloaded: int):
        """Update with new bytes downloaded."""
        current_time = time.time()
        self.bytes_window.append((current_time, bytes_downloaded))
        self.total_bytes += bytes_downloaded
        
        # Remove old entries
        while (self.bytes_window and
               current_time - self.bytes_window[0][0] > self.window_size):
            self.bytes_window.popleft()

    def get_speed(self) -> float:
        """Get current speed in bytes per second."""
        if not self.bytes_window:
            return 0.0
        
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Sum bytes in window
        bytes_in_window = sum(bytes_count for timestamp, bytes_count
                            in self.bytes_window
                            if timestamp > window_start)
        
        # Calculate time difference
        oldest_time = max(window_start,
                         self.bytes_window[0][0] if self.bytes_window else current_time)
        time_diff = current_time - oldest_time
        
        if time_diff <= 0:
            return 0.0
        
        return bytes_in_window / time_diff

    def get_average_speed(self) -> float:
        """Get average speed since start in bytes per second."""
        time_diff = time.time() - self.start_time
        if time_diff <= 0:
            return 0.0
        return self.total_bytes / time_diff

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
        if expected_time > elapsed:
            time.sleep(expected_time - elapsed)

    def update_limit(self, new_max_speed: float):
        """Update the rate limit."""
        self.max_speed = new_max_speed
        # Reset counters to avoid sudden speed changes
        self.downloaded = 0
        self.start_time = time.time()

@dataclass
class DownloadConfig:
    """Configuration for download operations."""
    num_threads: int = 4
    chunk_size: int = 1024 * 1024  # 1MB
    max_retries: int = 3
    connect_timeout: int = 10
    read_timeout: int = 30
    min_free_space_mb: int = 1000  # Minimum 1GB free space required
    speed_check_interval: int = 5  # seconds for progress updates
    verify_downloads: bool = False  # Verify existing downloads
    fix_broken: bool = False  # Remove and redownload corrupted files
    force_download: bool = False  # Force fresh download

    @classmethod
    def calculate_optimal_threads(cls) -> int:
        """Calculate optimal number of threads based on system capabilities.
        
        Always reserves:
        - 1 thread for Ctrl+C handling (unless only 1 thread available)
        - 1 thread kept free for system responsiveness
        """
        available_threads = multiprocessing.cpu_count()
        
        if available_threads == 1:
            # If only 1 thread, use it (no Ctrl+C handler possible)
            return 1
        elif available_threads == 2:
            # With 2 threads, use 1 for download and 1 for Ctrl+C
            return 1
        else:
            # Keep one thread for Ctrl+C and one free, use the rest
            return max(1, available_threads - 2)

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # First, adjust based on CPU
        available_threads = multiprocessing.cpu_count()
        optimal_threads = self.calculate_optimal_threads()
        
        # Log thread configuration
        logger.info(f"System capabilities:")
        logger.info(f"- CPU threads: {available_threads}")
        logger.info(f"- Optimal threads for download: {optimal_threads}")
        
        # Adjust thread count if needed
        if self.num_threads >= available_threads:
            logger.warning(
                f"Requested {self.num_threads} threads exceeds system capacity ({available_threads} threads). "
                f"Adjusting to {optimal_threads} threads:"
            )
            logger.info(f"- Reserved: 1 thread for Ctrl+C handling")
            logger.info(f"- Reserved: 1 thread for system responsiveness")
            logger.info(f"- Available: {optimal_threads} threads for downloads")
            self.num_threads = optimal_threads
        else:
            logger.info(f"Using requested thread count: {self.num_threads}")

class DownloadManager:
    @staticmethod
    def _normalize_repo_id(repo_id_or_url: str) -> str:
        """Normalize repository ID from URL or direct input."""
        repo_id = repo_id_or_url.replace("https://huggingface.co/", "")
        return repo_id.rstrip("/")

    @staticmethod
    def _get_auth_token() -> Optional[str]:
        """Safely retrieve authentication token."""
        try:
            cache_info = scan_cache_dir()
            return getattr(cache_info, 'token', None)  # Return None if no token
        except Exception as e:
            logger.debug(f"No auth token available: {e}")  # Debug level since this is ok for public repos
            return None

    def _signal_handler(self, sig, frame):
        """Handle interrupt signal."""
        # Only handle Ctrl+C if we have more than 1 thread
        if multiprocessing.cpu_count() > 1:
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
        else:
            # With only 1 thread, we can't handle Ctrl+C gracefully
            logger.warning("System has only 1 thread available. Use Ctrl+Break or Task Manager to force exit if needed.")

    def __init__(self, repo_id: str, output_dir: str, repo_type: str = "model",
                 config: Optional[DownloadConfig] = None):
        self.repo_id = self._normalize_repo_id(repo_id)
        self.model_name = self.repo_id.split("/")[-1]
        self.output_dir = Path(output_dir) / self.model_name
        self.repo_type = repo_type
        self.config = config or DownloadConfig()
        self.exit_event = threading.Event()
        self.completed_files: Set[str] = set()
        self.download_queue = queue.Queue()
        self.file_locks = {}
        self.token = None  # Initialize without token, will get if needed
        self.speed_trackers: Dict[str, SpeedTracker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.total_speed_tracker = SpeedTracker()
        self.download_state: Optional[DownloadState] = None
        
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

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

            # Use file lock to prevent concurrent access
            if filename not in self.file_locks:
                self.file_locks[filename] = threading.Lock()

            with self.file_locks[filename]:
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
        """Download with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(url, stream=stream, headers=headers,
                                     timeout=timeout, verify=True)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Request failed: {e}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    def _check_disk_space(self, required_mb: int) -> bool:
        """Check if there's enough disk space available."""
        try:
            total, used, free = shutil.disk_usage(self.output_dir.parent)
            free_mb = free // (1024 * 1024)
            return free_mb >= required_mb
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return False

    def _verify_file_integrity(self, file_path: Path, expected_size: int) -> bool:
        """Verify downloaded file integrity."""
        try:
            actual_size = file_path.stat().st_size
            return actual_size == expected_size
        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False

    def _parse_lfs_pointer(self, pointer_content: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse Git LFS pointer file."""
        match = re.match(
            r"version https://git-lfs\.github\.com/spec/v1\noid sha256:([a-f0-9]+)\nsize (\d+)",
            pointer_content,
        )
        if match:
            return match.group(1), int(match.group(2))
        return None, None

    def _download_lfs_file(self, filename: str, local_path: Path) -> bool:
        """Download LFS file with resume capability."""
        try:
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
                        headers["Range"] = f"bytes={existing_size}-"
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

    def download(self) -> bool:
        """Main download method with state tracking and verification."""
        try:
            # Try to access repository (first without token, then with if needed)
            repo_info = self._try_repo_access()
            if not repo_info:
                return False
                
            # Store token if we needed it for access
            self.token = getattr(repo_info, '_token', None)

            # Get optimal thread count based on system capabilities
            optimal_threads = self._get_optimal_threads()
            self.config.num_threads = optimal_threads

            # Get file list
            files = [(file.rfilename, file.size, file.lfs is not None)
                    for file in repo_info.siblings]
            
            try:
                # Initialize or load state
                self._initialize_state(files)
            except Exception as e:
                logger.warning(f"Failed to initialize state: {e}")
                logger.info("Creating fresh state and checking existing files...")
                
                # Create output directory
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create new state without trying to load existing
                self.download_state = DownloadState.create_new(self.repo_id, files)
                self.download_state.set_state_file(self.output_dir / '.download_state')
                
                # Always verify files when state initialization failed
                logger.info("Verifying any existing files...")
                self._verify_existing_files()
                
                # Try to save new state
                try:
                    self.download_state.save()
                except Exception as save_error:
                    logger.warning(f"Could not save state file: {save_error}")
                    logger.info("Continuing without state persistence...")
            
            # Prepare download queue (skip completed files)
            for filename, size, is_lfs in files:
                if filename not in self.completed_files:
                    self.download_queue.put((filename, is_lfs))

            if self.download_queue.empty():
                logger.info("All files already downloaded and verified")
                return True

            logger.info(f"Found {len(files)} files in repository '{self.repo_id}'")
            logger.info(f"Downloading to: {self.output_dir}")

            # Start with CPU-based thread count
            current_threads = self.config.num_threads

            # Start download threads
            with ThreadPoolExecutor(max_workers=current_threads) as executor:
                workers = [executor.submit(self._worker) for _ in range(current_threads)]
                
                # Monitor download progress
                while not self.download_queue.empty() and not self.exit_event.is_set():
                    time.sleep(self.config.speed_check_interval)
                    current_speed = self.total_speed_tracker.get_speed()
                    avg_speed = self.total_speed_tracker.get_average_speed()
                    active_threads = len([t for t in self.speed_trackers.values() if t.get_speed() > 0])
                    
                    # Log download progress periodically
                    if not hasattr(self, '_last_logged_download_speed') or \
                       abs(current_speed - getattr(self, '_last_logged_download_speed', 0)) / max(current_speed, 1) > 0.2:
                        self._last_logged_download_speed = current_speed
                        logger.info(
                            f"Download speed: {current_speed / (1024*1024):.2f} MB/s "
                            f"(avg: {avg_speed / (1024*1024):.2f} MB/s) "
                            f"using {active_threads}/{current_threads} threads"
                        )

            # Wait for completion or interruption
            try:
                self.download_queue.join()
                
                if self.exit_event.is_set():
                    # Save final state before exiting
                    if self.download_state:
                        try:
                            self.download_state.save()
                            logger.info("Download state saved successfully")
                        except Exception as e:
                            logger.error(f"Failed to save download state: {e}")
                    
                    # Calculate overall progress
                    total_files = len(self.download_state.files)
                    completed = len(self.completed_files)
                    in_progress = len([f for f in self.download_state.files
                                     if self.download_state.files[f]['status'] == 'in_progress'])
                    
                    logger.info(
                        f"Download interrupted: {completed}/{total_files} files completed "
                        f"({completed/total_files*100:.1f}%), {in_progress} files in progress"
                    )
                    return False

                logger.info("Download completed successfully")
                return True
                
            except KeyboardInterrupt:
                # Handle any direct KeyboardInterrupt that might bypass signal handler
                logger.info("Received keyboard interrupt. Stopping downloads...")
                self.exit_event.set()
                # Let the signal handler do the cleanup
                return False

        except Exception as e:
            logger.error(f"Download failed: {e}")
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
    
    args = parser.parse_args()

    if not args.repo_id_or_url:
        args.repo_id_or_url = input("Enter the Hugging Face repository ID or URL: ")

    config = DownloadConfig(
        num_threads=args.threads,
        chunk_size=args.chunk_size,
        min_free_space_mb=args.min_free_space,
        verify_downloads=args.verify,
        fix_broken=args.fix_broken,
        force_download=args.force
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
