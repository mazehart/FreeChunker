import uuid
from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
import os
import gc
from src.sentenizer import Sentenceizer

input_dir = './ThePile/8k'  # Changed to 8k, safer test
output_dir = './ThePile_8k_sentences'
temp_dir = './temp_batches'

# Batch processing parameters
BATCH_SIZE = 1000

# Initialize sentence splitter (using optimized version)
print("Initializing optimized Sentenceizer for sentence splitting...")
sentenceizer = Sentenceizer()

def split_sentences_only(text):
    """Split sentences using optimized Sentenceizer."""
    sentences = sentenceizer.split(text)
    
    # Filter empty sentences
    selected = []
    for sent in sentences:
        if sent.strip():  # Skip empty sentences
            selected.append(sent.strip())
    
    return selected

# Create temporary directory
os.makedirs(temp_dir, exist_ok=True)

# Load dataset
print(f"Loading dataset from {input_dir}...")
dataset = load_from_disk(input_dir)
total_records = len(dataset)
print(f"Dataset contains {total_records} records")
print(f"Processing in {(total_records + BATCH_SIZE - 1) // BATCH_SIZE} batches, {BATCH_SIZE} records per batch")

batch_files = []
current_batch = []
processed_count = 0
batch_count = 0

# Process data in batches
for i, item in enumerate(tqdm(dataset, desc='Batch processing text and splitting sentences')):
    text = item.get('text', '')
    if not text.strip():  # Skip empty text
        continue
        
    sentences = split_sentences_only(text)
    if sentences:  # Only save records with sentences
        current_batch.append({
            'uuid': item.get('uuid', str(uuid.uuid4())),
            'sentences': sentences,
            'source': item.get('source', ''),
            'original_token_count': item.get('token_count', 0)
        })
        processed_count += 1
    
    # Save current batch when batch size is reached or all data processed
    if len(current_batch) >= BATCH_SIZE or i == total_records - 1:
        if current_batch:  # Ensure batch is not empty
            batch_count += 1
            batch_file = f"{temp_dir}/batch_{batch_count:04d}"
            
            print(f"\nSaving batch {batch_count}, containing {len(current_batch)} records...")
            batch_dataset = Dataset.from_list(current_batch)
            batch_dataset.save_to_disk(batch_file)
            batch_files.append(batch_file)
            
            # Clean up memory
            del current_batch, batch_dataset
            current_batch = []
            gc.collect()
            
            print(f"Batch {batch_count} saved to {batch_file}")

print(f"\n✅ All batches processed!")
print(f"Total processed {processed_count} valid records, divided into {len(batch_files)} batches")

# Merge all batches
print(f"\nStarting to merge {len(batch_files)} batches...")
datasets_to_merge = []

for i, batch_file in enumerate(tqdm(batch_files, desc="Loading batches for merging")):
    batch_dataset = load_from_disk(batch_file)
    datasets_to_merge.append(batch_dataset)
    
    # If too many datasets accumulated, perform partial merge first
    if len(datasets_to_merge) >= 10:
        print(f"Performing intermediate merge (loaded {i+1}/{len(batch_files)} batches)...")
        merged_partial = concatenate_datasets(datasets_to_merge)
        datasets_to_merge = [merged_partial]
        gc.collect()

# Final merge
print("Performing final merge...")
final_dataset = concatenate_datasets(datasets_to_merge)

# Save final dataset
print(f"Saving final dataset to {output_dir}...")
os.makedirs(output_dir, exist_ok=True)
final_dataset.save_to_disk(output_dir)

# Clean up temporary files
print("Cleaning up temporary files...")
import shutil
shutil.rmtree(temp_dir)

print(f"\n🎉 Dataset construction completed!")
print(f"Final dataset contains {len(final_dataset)} records")
print(f"Saved to: {output_dir}")

# Show statistics
total_sentences = sum(len(record['sentences']) for record in final_dataset)
print(f"Total sentences: {total_sentences}")
print(f"Average sentences per document: {total_sentences / len(final_dataset):.2f}")