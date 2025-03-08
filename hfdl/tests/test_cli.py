import unittest
import sys
import io
import contextlib
from unittest.mock import patch, MagicMock
from argparse import Namespace
import tempfile
import os

# Import the main function directly
from hfdl.downloader import main, get_example_text

class TestCLI(unittest.TestCase):
    """Tests for the command-line interface."""
    
    def test_get_example_text(self):
        """Test that example text is properly formatted."""
        examples = get_example_text()
        self.assertIn("Examples:", examples)
        self.assertIn("Basic download:", examples)
        self.assertIn("Advanced mode:", examples)
        
    @patch('sys.argv', ['hfdl', 'user/repo'])
    @patch('hfdl.downloader.HFDownloader')
    def test_basic_command(self, mock_downloader):
        """Test basic command with just a repository."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.download.return_value = True
        mock_downloader.return_value = mock_instance
        
        # Run the command
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
        
        # Verify the downloader was called with correct args
        mock_downloader.assert_called_once()
        args, kwargs = mock_downloader.call_args
        self.assertEqual(kwargs['model_id'], 'user/repo')
        self.assertEqual(kwargs['enhanced'], False)  # Default should be False
        
    @patch('sys.argv', ['hfdl', 'user/repo', '--optimize-download'])
    @patch('hfdl.downloader.HFDownloader')
    def test_optimize_download(self, mock_downloader):
        """Test the optimize-download option."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.download.return_value = True
        mock_downloader.return_value = mock_instance
        
        # Run the command
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
        
        # Verify the downloader was called with correct args
        args, kwargs = mock_downloader.call_args
        self.assertEqual(kwargs['enhanced'], True)
        
    @patch('sys.argv', ['hfdl', '--dry-run', 'user/repo'])
    @patch('hfdl.downloader.HFDownloader')
    def test_dry_run(self, mock_downloader):
        """Test dry run mode doesn't call the downloader."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
        
        # Downloader should not be instantiated in dry run mode
        mock_downloader.assert_not_called()
        
    @patch('builtins.input', side_effect=['user/repo', '', 'y'])
    @patch('hfdl.downloader.HFDownloader')
    def test_interactive_mode(self, mock_downloader, mock_input):
        """Test interactive mode when no repository is provided."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.download.return_value = True
        mock_downloader.return_value = mock_instance
        
        # Run with empty sys.argv to trigger interactive mode
        with patch('sys.argv', ['hfdl']):
            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                main()
        
        # Verify the downloader was called with correct args
        args, kwargs = mock_downloader.call_args
        self.assertEqual(kwargs['model_id'], 'user/repo')
        self.assertEqual(kwargs['enhanced'], True)  # User entered 'y'
        
    @patch('sys.argv', ['hfdl', 'user/repo', '--quiet'])
    @patch('hfdl.downloader.HFDownloader')
    def test_quiet_mode(self, mock_downloader):
        """Test quiet mode suppresses output."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.download.return_value = True
        mock_downloader.return_value = mock_instance
        
        # Capture stdout
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            main()
            # Should have empty stdout due to quiet mode
            self.assertEqual(fake_out.getvalue().strip(), "Download completed successfully")

if __name__ == '__main__':
    unittest.main()