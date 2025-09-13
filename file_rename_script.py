import os
import glob

# Set your folder path
folder_path = "/home/student/pc_deploy/Semester_2/BikeSafeAI/fatbikes"

# Get all image files (adjust extensions as needed)
image_files = sorted(glob.glob(os.path.join(folder_path, "*.*")), key=os.path.getmtime)

# Rename images sequentially
for index, file in enumerate(image_files, start=1):
    ext = os.path.splitext(file)[1]  # Get file extension
    new_name = f"fatbike_image_{index:03d}{ext}"  # Format: image_001.jpg
    new_path = os.path.join(folder_path, new_name)

    os.rename(file, new_path)

print("Renaming completed.")
