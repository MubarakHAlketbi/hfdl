import os
import platform
import unittest
from pathlib import Path
from hfdl.utils import sanitize_filename, get_os_compatible_path

class TestUtils(unittest.TestCase):
    """Test utilities for OS-agnostic path handling"""
    
    def test_sanitize_filename_iso_timestamp(self):
        """Test sanitization of ISO timestamps in filenames"""
        # Input with ISO timestamp with colons
        filename = "2025-03-04T06:55:30.344558.log.lock"
        
        # Expected result depends on platform
        if platform.system() == "Windows":
            expected = "2025-03-04T06_55_30.344558.log.lock"
        else:
            # On Unix-like systems, colons are allowed in filenames
            expected = filename
            
        self.assertEqual(sanitize_filename(filename), expected)
    
    def test_sanitize_filename_with_colons(self):
        """Test sanitization of general filenames with colons"""
        # Input with colons
        filename = "test:file:name.txt"
        
        # Expected result depends on platform
        if platform.system() == "Windows":
            expected = "test_file_name.txt"
        else:
            # On Unix-like systems, colons are allowed in filenames
            expected = filename
            
        self.assertEqual(sanitize_filename(filename), expected)
    
    def test_get_os_compatible_path(self):
        """Test path compatibility conversion"""
        # Create a test path with problematic characters
        test_path = os.path.join("downloads", "model", "2025-03-04T06:55:30.cache", "file.txt")
        
        result = get_os_compatible_path(test_path)
        
        # Check that the path is normalized
        self.assertEqual(os.path.normpath(result), result)
        
        # On Windows, check that colons are replaced
        if platform.system() == "Windows":
            self.assertNotIn(":", result)
            
    def test_nested_paths(self):
        """Test sanitization of nested paths with multiple problematic components"""
        # Create a complex test path
        test_path = os.path.join(
            "downloads", 
            "model:name", 
            "2025-03-04T06:55:30.cache", 
            "subfolder:name",
            "file:name.txt"
        )
        
        result = get_os_compatible_path(test_path)
        
        # On Windows, verify all colons are gone
        if platform.system() == "Windows":
            self.assertNotIn(":", result)
            
        # Verify path structure is preserved (same number of components)
        self.assertEqual(len(test_path.split(os.sep)), len(result.split(os.sep)))

if __name__ == "__main__":
    unittest.main()