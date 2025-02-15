import argparse
import re
import sys
from .downloader import HFDownloader
from hfdl import __version__

def validate_model_id(value):
    """Validate repository ID format."""
    # Remove URL prefix if present
    repo_id = value.replace("https://huggingface.co/", "").rstrip("/")
    
    if not re.match(r'^[\w.-]+/[\w.-]+$', repo_id):
        raise argparse.ArgumentTypeError(
            f"Invalid repository ID format: {repo_id}\n"
            "Expected format: username/repository-name\n"
            "Allowed characters: letters, numbers, hyphens, underscores, and dots"
        )
    return repo_id

def validate_threads(value):
    """Validate thread count, allowing 'auto' or positive integers."""
    if value.lower() == 'auto':
        return 0  # 0 means auto in our implementation
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

def main():
    parser = argparse.ArgumentParser(
        description='Hugging Face Downloader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Version argument
    parser.add_argument('-v', '--version',
                       action='version',
                       version=f'%(prog)s {__version__}')
    
    # Model ID argument
    parser.add_argument('model_id', nargs='?', type=validate_model_id,
                       help='Model/dataset identifier (username/modelname)')
    
    # Basic options
    parser.add_argument('-d', '--directory', default='downloads',
                      help='Base download directory')
    parser.add_argument('-t', '--threads', type=validate_threads, default='auto',
                      help='Number of download threads (auto or positive integer)')
    parser.add_argument('-r', '--repo-type', choices=['model', 'dataset', 'space'],
                      default='model', help='Repository type')
    
    # Download options
    parser.add_argument('--verify', action='store_true',
                      help='Verify downloads')
    parser.add_argument('--force', action='store_true',
                      help='Force fresh download')
    parser.add_argument('--no-resume', action='store_true',
                      help='Disable download resuming')
    
    args = parser.parse_args()
    
    try:
        downloader = HFDownloader(
            model_id=args.model_id,
            download_dir=args.directory,
            num_threads=args.threads,  # 0 means auto
            repo_type=args.repo_type,
            verify=args.verify,
            force=args.force,
            resume=not args.no_resume
        )
        success = downloader.download()
        if success:
            print("\nDownload completed successfully")
        else:
            print("\nDownload completed with errors", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()