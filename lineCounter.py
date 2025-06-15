import os
import argparse
from pathlib import Path


def count_lines_in_file(file_path):
    """Count the number of lines in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return sum(1 for line in file)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def count_python_lines(folder_path, include_subdirs=True):
    """
    Count total lines of Python code in a folder.
    
    Args:
        folder_path (str): Path to the folder to analyze
        include_subdirs (bool): Whether to include subdirectories
    
    Returns:
        tuple: (total_lines, file_count, file_details)
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory")
    
    total_lines = 0
    file_count = 0
    file_details = []
    
    # Define the pattern for searching
    pattern = "**/*.py" if include_subdirs else "*.py"
    
    # Find all Python files
    for py_file in folder_path.glob(pattern):
        if py_file.is_file():
            lines = count_lines_in_file(py_file)
            total_lines += lines
            file_count += 1
            file_details.append((str(py_file.relative_to(folder_path)), lines))
    
    return total_lines, file_count, file_details


def main():
    parser = argparse.ArgumentParser(description="Count lines of Python code in a folder")
    parser.add_argument("folder", nargs='?', default=".", 
                       help="Folder path to analyze (default: current directory)")
    parser.add_argument("--no-subdirs", action="store_true", 
                       help="Don't include subdirectories")
    parser.add_argument("--details", action="store_true", 
                       help="Show details for each file")
    
    args = parser.parse_args()
    
    try:
        total_lines, file_count, file_details = count_python_lines(
            args.folder, 
            include_subdirs=not args.no_subdirs
        )
        
        print(f"\n📊 Python Code Line Count Summary")
        print(f"{'='*40}")
        print(f"Folder analyzed: {os.path.abspath(args.folder)}")
        print(f"Total Python files: {file_count}")
        print(f"Total lines of code: {total_lines:,}")
        
        if args.details and file_details:
            print(f"\n📄 File Details:")
            print(f"{'='*40}")
            file_details.sort(key=lambda x: x[1], reverse=True)  # Sort by line count
            for file_path, lines in file_details:
                print(f"{lines:>6,} lines - {file_path}")
        
        if file_count == 0:
            print("\n⚠️  No Python files found in the specified folder.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
