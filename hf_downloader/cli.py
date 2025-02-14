import argparse
import sys
from .downloader import HFDownloader

def validate_model_id(value):
    if "/" not in value:
        raise argparse.ArgumentTypeError("Model ID must be in format username/modelname")
    return value

def main():
    parser = argparse.ArgumentParser(description='Hugging Face Downloader', 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required arguments
    parser.add_argument('model_id', type=validate_model_id,
                      help='Model/dataset identifier (username/modelname)')
    
    # Directory and threading
    parser.add_argument('-d', '--directory', default='downloads',
                      help='Base download directory')
    parser.add_argument('-t', '--threads', type=int, default=0,
                      help='0 for auto-detect (default: %(default)s)')
    
    # Repository options
    parser.add_argument('-r', '--repo-type', choices=['model', 'dataset', 'space'],
                      default='model', help='Repository type')
    parser.add_argument('--chunk-size', type=int, default=1024*1024,
                      help='Download chunk size in bytes')
    
    # Verification options
    parser.add_argument('--verify', action='store_true',
                      help='Verify existing downloads')
    parser.add_argument('--fix-broken', action='store_true',
                      help='Remove and redownload corrupted files')
    
    # System options
    parser.add_argument('--min-free-space', type=int, default=5000,
                      help='Minimum required free space in MB')
    parser.add_argument('--file-size-threshold', type=int, default=200,
                      help='Size threshold for big files in MB')
    
    # Performance tuning
    parser.add_argument('--min-speed-percentage', type=int, default=5,
                      choices=range(1, 101), metavar="1-100",
                      help='Min speed percentage per thread')
    parser.add_argument('--speed-test-duration', type=int, default=5,
                      help='Speed test duration in seconds')
    
    # Operation modes
    parser.add_argument('--force', action='store_true',
                      help='Force fresh download, ignore existing files')
    
    args = parser.parse_args()
    
    try:
        downloader = HFDownloader(
            args.model_id,
            download_dir=args.directory,
            num_threads=args.threads or 'auto',
            repo_type=args.repo_type,
            chunk_size=args.chunk_size,
            min_free_space=args.min_free_space,
            file_size_threshold=args.file_size_threshold,
            min_speed_percentage=args.min_speed_percentage,
            speed_test_duration=args.speed_test_duration,
            verify=args.verify,
            fix_broken=args.fix_broken,
            force=args.force
        )
        downloader.download()
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        print("\nOperation completed")

if __name__ == "__main__":
    main()