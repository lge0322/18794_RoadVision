import os
import shutil

gtcoarse_dir = 'dataset_original/gtCoarse'
gtfine_dir = 'dataset_original/gtFine'

def extract_json_files(source_dir, output_dir):
    # Walk through each subdirectory (train/val/cityname)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)  # Get the relative path from the source directory
                target_dir = os.path.join(output_dir, os.path.basename(source_dir), relative_path)  # Add gtCoarse or gtFine to path
                os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists
                
                shutil.copy(file_path, target_dir)
                print(f'Copied: {file_path} to {target_dir}')

extract_json_files(os.path.join(gtcoarse_dir, 'train'), "dataset/gtCoarse/train")
extract_json_files(os.path.join(gtcoarse_dir, 'val'), "dataset/gtCoarse/val")
extract_json_files(os.path.join(gtfine_dir, 'train'), "dataset/gtFine/train")
extract_json_files(os.path.join(gtfine_dir, 'val'), "dataset/gtFine/val")
