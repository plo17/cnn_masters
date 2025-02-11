import os

def rename_files_based_on_folder(directory_path):
    try:
        # Check if the provided path exists and is a directory
        if not os.path.isdir(directory_path):
            print("The provided path is not a directory.")
            return

        # Iterate through all subdirectories and files in the main directory
        for root, dirs, files in os.walk(directory_path):
            # Name of the current folder
            folder_name = os.path.basename(root)

            # Process only directories that contain files
            if files:
                print(f"Processing directory: {root}")

                for index, filename in enumerate(files, start=1):
                    # Split the file name into name and extension
                    _, extension = os.path.splitext(filename)
                    # Create a new file name based on the folder name and numbering
                    new_name = f"{folder_name}_{index:03d}{extension}"
                    # Rename the file
                    os.rename(
                        os.path.join(root, filename),
                        os.path.join(root, new_name)
                    )
                    print(f"Renamed: {filename} -> {new_name}")
            else:
                print(f"The directory {root} does not contain any files.")
    except Exception as e:
        print(f"An error occurred: {e}")



directory_path = input("Enter the path to the main directory: ")
rename_files_based_on_folder(directory_path)
