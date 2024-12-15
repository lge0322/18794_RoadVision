import os
import re

base_path = './dataset/leftImg8bit'
splits = ['test', 'train', 'val']

counter = 0
for split in splits:
    split_path = os.path.join(base_path, split)

    for city_name in os.listdir(split_path):
        city_path = os.path.join(split_path, city_name)
        
        # Ensure only accessing directories
        if not os.path.isdir(city_path):
            continue
        
        for file in os.listdir(city_path):
            if not file.endswith('.png'):
                continue
            try:
                newname = re.sub(r'_leftImg8bit', '', file)
            except IndexError as e:
                print(f"Skipping file {file} in {city_path}: {e}")
                continue
            
            src_file = os.path.join(city_path, file)
            dst_file = os.path.join(city_path, newname)
            os.rename(src_file, dst_file)
            print("Successfully renamed {} -> {}".format(src_file, dst_file))
            counter += 1

print(f"Total files renamed: {counter}")
