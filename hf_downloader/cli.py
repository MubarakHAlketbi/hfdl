import argparse
from .downloader import HFDownloader

def main():
    parser = argparse.ArgumentParser(description='Hugging Face Downloader')
    parser.add_argument('model_id', help='Model identifier (username/modelname)')
    parser.add_argument('-d', '--directory', default='downloads')
    parser.add_argument('-t', '--threads', type=int)
    # Add other CLI arguments
    
    args = parser.parse_args()
    
    downloader = HFDownloader(
        args.model_id,
        download_dir=args.directory,
        num_threads=args.threads
    )
    downloader.download()

if __name__ == "__main__":
    main()
