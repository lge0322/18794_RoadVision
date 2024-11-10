import os
import re

# Define the base path
base_path = './dataset/leftImg8bit'
# Define the split folders
splits = ['test', 'train', 'val']

counter = 0
for split in splits:
    split_path = os.path.join(base_path, split)
    # Go through each city folder within the current split folder
    for city_name in os.listdir(split_path):
        city_path = os.path.join(split_path, city_name)
        # Ensure we're only accessing directories (skip files)
        if not os.path.isdir(city_path):
            continue
        # Process each file in the city folder
        for file in os.listdir(city_path):
            # Ensure we're only working with .png files
            if not file.endswith('.png'):
                continue
            try:
                # Extract the new name based on the pattern
                newname = re.sub(r'_leftImg8bit', '', file)
            except IndexError as e:
                print(f"Skipping file {file} in {city_path}: {e}")
                continue
            
            # Construct the source and destination file paths
            src_file = os.path.join(city_path, file)
            dst_file = os.path.join(city_path, newname)
            # Rename the file
            os.rename(src_file, dst_file)
            print("Successfully renamed {} -> {}".format(src_file, dst_file))
            counter += 1

print(f"Total files renamed: {counter}")
