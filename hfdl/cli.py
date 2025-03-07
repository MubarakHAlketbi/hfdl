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
    except (ValueError, argparse.ArgumentTypeError) as e:
        # Handle validation errors
        print(f"\nValidation error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        # Handle file system errors
        print(f"\nFile not found: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        # Handle permission errors
        print(f"\nPermission denied: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        # Handle import errors
        print(f"\nImport error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle user interruption
        print("\nDownload cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"\nUnexpected error: {str(e)}", file=sys.stderr)
        # Print more details in debug mode
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()