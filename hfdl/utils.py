import os
import re
import platform

def sanitize_filename(filename):
    """
    Sanitize a filename to be valid across different operating systems.
    Particularly handles Windows restrictions on characters like colons in filenames.
    
    Args:
        filename (str): The filename to sanitize
        
    Returns:
        str: A sanitized filename that is valid across operating systems
    """
    # Characters not allowed in Windows filenames: \ / : * ? " < > |
    # Replace colons with underscores specifically (main issue with timestamps)
    if platform.system() == "Windows":
        # Replace colons in ISO timestamps (like 2025-03-04T06:55:30)
        filename = re.sub(r'(\d{4}-\d{2}-\d{2}T\d{2})(:)(\d{2})(:)(\d{2})', r'\1_\3_\5', filename)
        # Replace any remaining colons
        filename = filename.replace(':', '_')
    
    return filename

def get_os_compatible_path(path):
    """
    Ensures a path is compatible with the current operating system.
    Handles directory separators and filename restrictions.
    
    Args:
        path (str): The path to make compatible
        
    Returns:
        str: An OS-compatible path
    """
    # Normalize path separators based on OS
    path = os.path.normpath(path)
    
    # Split path into components
    parts = path.split(os.sep)
    
    # Sanitize each filename in the path
    sanitized_parts = [sanitize_filename(part) for part in parts]
    
    # Rejoin with appropriate separator
    return os.sep.join(sanitized_parts)