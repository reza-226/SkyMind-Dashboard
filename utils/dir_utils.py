import os


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def ensure_file_dir(filepath: str):
    """
    Ensure directory for a file path exists.
    Example: ensure_file_dir("results/easy/run1/log.csv")
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return filepath
