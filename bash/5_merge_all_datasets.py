from datasets import load_from_disk, concatenate_datasets
import os
import glob
from tqdm import tqdm

def merge_all_parts():
    """Merge all dataset parts"""
    output_dir = '/share/home/ecnuzwx/UnifiedRAG/vector/jina-embeddings-v2-small-en'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating output directory {output_dir}")
    
    # Find all part files
    part_pattern = f"{output_dir}/part_*"
    part_dirs = glob.glob(part_pattern)
    
    if not part_dirs:
        print(f"Error: No part files found in {output_dir}")
        return
    
    # Sort by part number
    part_dirs.sort(key=lambda x: int(x.split('_')[-1]))
    
    print(f'Found {len(part_dirs)} part files:')
    for part_dir in part_dirs:
        print(f'  - {part_dir}')
    
    print('\nStarting to merge all parts...')
    all_parts = []
    
    for part_dir in tqdm(part_dirs, desc='Loading parts'):
        part_ds = load_from_disk(part_dir)
        all_parts.append(part_ds)
        print(f'Loaded {part_dir}, contains {len(part_ds)} records')
    
    if not all_parts:
        print("Error: No parts loaded successfully")
        return
    
    print('\nMerging datasets...')
    full_ds = concatenate_datasets(all_parts)
    
    print(f'\nSaving full dataset to {output_dir}...')
    full_ds.save_to_disk(output_dir)
    
    print(f'Merge completed!')
    print(f'  Total records: {len(full_ds)}')
    print(f'  Saved path: {output_dir}')

if __name__ == "__main__":
    merge_all_parts()
