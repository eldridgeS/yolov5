import os
import shutil
import re

def sort_files_by_number_range(source_folder, destination_folder, min_num=201, max_num=250):
    """
    Looks at a folder's contents, and if a file's name contains a three-digit
    number within a specific range (001-250), it is copied into another folder.

    Args:
        source_folder (str): The path to the folder containing the files to be scanned.
        destination_folder (str): The path to the folder where matching files will be copied.
        min_num (int): The minimum three-digit number (inclusive) in the range. Default is 1.
        max_num (int): The maximum three-digit number (inclusive) in the range. Default is 250.
    """

    # Ensure source folder exists
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: '{destination_folder}'")

    # Regular expression to find a three-digit number in the filename
    # \d{3} matches exactly three digits
    # It will find the first occurrence if multiple three-digit numbers exist
    number_pattern = re.compile(r'\d{3}')

    print(f"Scanning files in: '{source_folder}'")
    print(f"Looking for files with numbers between {min_num:03d} and {max_num:03d}")

    copied_count = 0
    skipped_count = 0

    for filename in os.listdir(source_folder):
        # Construct full file path
        source_file_path = os.path.join(source_folder, filename)

        # Skip directories
        if os.path.isdir(source_file_path):
            continue

        match = number_pattern.search(filename)
        if match:
            try:
                found_number_str = match.group(0)
                found_number = int(found_number_str)

                if min_num <= found_number <= max_num:
                    destination_file_path = os.path.join(destination_folder, filename)
                    try:
                        shutil.copy2(source_file_path, destination_file_path)
                        print(f"Copied '{filename}' (found number: {found_number_str}) to '{destination_folder}'")
                        copied_count += 1
                    except IOError as e:
                        print(f"Error copying '{filename}': {e}")
                else:
                    # print(f"Skipped '{filename}' (number {found_number_str} out of range)")
                    skipped_count += 1
            except ValueError:
                print(f"Warning: Could not convert '{found_number_str}' in '{filename}' to an integer. Skipping.")
                skipped_count += 1
        else:
            # print(f"Skipped '{filename}' (no three-digit number found)")
            skipped_count += 1

    print("-" * 30)
    print(f"Process complete.")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped: {skipped_count}")

# --- Configuration ---
# IMPORTANT: Replace these with your actual folder paths
SOURCE_DIR = '1_A'  # e.g., 'C:/Users/YourUser/Documents/MyFiles' or '/home/youruser/my_files'
DESTINATION_DIR = '1500_Set' # e.g., 'C:/Users/YourUser/Documents/FilteredFiles' or '/home/youruser/filtered_files'

# --- Run the script ---
if __name__ == "__main__":
    # Example usage (uncomment and modify paths to test)
    # To run this script, you must first create the 'source_folder' and 'destination_folder'
    # on your system, and place some dummy files in 'source_folder' with names like:
    # "document_045.txt", "image_200.jpg", "report_300.pdf", "file_001_v2.docx", "another_file.log"

    print("Please configure SOURCE_DIR and DESTINATION_DIR in the script before running.")
    print("Example: ")
    print("SOURCE_DIR = 'temp_source'")
    print("DESTINATION_DIR = 'temp_destination'")
    print("\nThen, create these folders and add some files for testing.")

    # Uncomment the following lines and change the paths for actual use:
    sort_files_by_number_range(SOURCE_DIR, DESTINATION_DIR, min_num=201, max_num=250)
