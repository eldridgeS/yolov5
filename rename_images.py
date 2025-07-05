import os
import re
import shutil

def rename_waste_images(root_dir, dry_run=True, subfolder_name_map=None):
    """
    Renames image files in the waste classification dataset to include their
    category and subfolder, ensuring unique and descriptive names.

    Args:
        root_dir (str): The path to the main 'images' directory of your dataset.
                        e.g., '/kaggle/input/recyclable-and-household-waste-classification/images/images'
        dry_run (bool): If True, only prints what *would* be done without
                        actually renaming files. Set to False to perform renaming.
        subfolder_name_map (dict, optional): A dictionary to map 'default'
                                            and 'real_world' to shorter names
                                            like {'default': 'def', 'real_world': 'rw'}.
                                            If None, uses full subfolder names.
    """
    if subfolder_name_map is None:
        subfolder_name_map = {} # Use empty map, so full names are used by default

    print(f"Starting image renaming process in: {root_dir}")
    print(f"Dry run mode: {dry_run}")
    if not dry_run:
        print("WARNING: This will permanently rename files. Ensure you have a backup.")

    # Counter for total renames
    renamed_count = 0

    # Walk through the directory structure
    for category_name in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_name)

        if not os.path.isdir(category_path):
            continue # Skip if it's not a directory

        # Sanitize category name for filename: lowercase and replace spaces/special chars
        category_slug = re.sub(r'[^a-z0-9]+', '_', category_name.lower())
        category_slug = category_slug.strip('_') # Remove leading/trailing underscores

        for subfolder_name in ['default', 'real_world']:
            subfolder_path = os.path.join(category_path, subfolder_name)

            if not os.path.isdir(subfolder_path):
                print(f"  Skipping non-existent subfolder: {subfolder_path}")
                continue # Skip if subfolder doesn't exist

            # Get the slug for the subfolder (e.g., 'def' for 'default')
            subfolder_slug = subfolder_name_map.get(subfolder_name, subfolder_name)

            # Get all image files in the subfolder
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            # Sort them to ensure consistent numbering
            image_files.sort(key=lambda x: int(re.search(r'_(\d+)\.', x).group(1)) if re.search(r'_(\d+)\.', x) else 0)

            print(f"  Processing: {category_name}/{subfolder_name} ({len(image_files)} images)")

            for i, old_filename in enumerate(image_files):
                # Extract original number (e.g., 1 from Image_1.png) and extension
                match = re.search(r'Image_(\d+)\.(.+)', old_filename, re.IGNORECASE)
                if not match:
                    print(f"    Skipping malformed filename: {old_filename} in {subfolder_path}")
                    continue

                original_number = int(match.group(1))
                extension = match.group(2)
                
                # Format the new filename
                # Example: plastic_water_bottles_def_001.png (padded to 3 digits)
                new_filename = f"{category_slug}_{subfolder_slug}_{original_number:03d}.{extension}"

                old_filepath = os.path.join(subfolder_path, old_filename)
                new_filepath = os.path.join(subfolder_path, new_filename)

                if dry_run:
                    print(f"    Dry run: Would rename '{old_filepath}' to '{new_filepath}'")
                else:
                    try:
                        shutil.move(old_filepath, new_filepath) # Use shutil.move for robustness
                        print(f"    Renamed '{old_filename}' to '{new_filename}'")
                        renamed_count += 1
                    except Exception as e:
                        print(f"    ERROR renaming '{old_filename}': {e}")
    
    print(f"\nRenaming process complete. Total files renamed: {renamed_count}")
    if dry_run:
        print("Note: No files were actually changed (dry_run=True). Set dry_run=False to apply changes.")

# --- How to use the script ---

if __name__ == "__main__":
    # IMPORTANT: Set this to the actual path of your 'images' directory
    # For example, if your dataset is unzipped at /home/user/waste_dataset/images/images
    # then root_directory_of_images = '/home/user/waste_dataset/images/images'
    # Or if it's within your Kaggle environment as per the description:
    root_directory_of_images = 'images'
    
    # Define how you want 'default' and 'real_world' subfolders to be named
    # For example: {'default': 'def', 'real_world': 'rw'} for shorter names
    # Or use {} for full names: {'default': 'default', 'real_world': 'real_world'}
    subfolder_map = {'default': 'def', 'real_world': 'rw'} # As per your example 'aerosol_cans_def1'

    # --- DRY RUN FIRST ---
    # It's highly recommended to run in dry_run=True mode first to see what changes will be made.
    print("\n--- Performing a DRY RUN first ---")
    rename_waste_images(root_directory_of_images, dry_run=True, subfolder_name_map=subfolder_map)

    # --- THEN, PERFORM ACTUAL RENAMING ---
    # After you are satisfied with the dry run output, change dry_run to False
    # and run the script again to apply the changes.
    # Make sure you have a backup of your data before running with dry_run=False!
    input("\nPress Enter to proceed with actual renaming (make sure you've backed up your data)...")
    print("\n--- Performing ACTUAL RENAMING ---")
    rename_waste_images(root_directory_of_images, dry_run=False, subfolder_name_map=subfolder_map)