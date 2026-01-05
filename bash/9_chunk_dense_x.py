from baseline.dense_x_retrieval import DenseXRetrievalChunking
from datasets import load_from_disk, Dataset, DatasetDict
from utils.monitor import Monitor
import time

# Initialize monitor
monitor = Monitor(device_id="4")  # Use GPU 4

# Setup environment
monitor.setup()
model_name = 'propositionizer-wiki-flan-t5-large'
# Initialization
print("🚀 Initializing...")
chunker = DenseXRetrievalChunking(model_name=model_name, device="cuda")

# Load all datasets
datasets = load_from_disk('./datasets')
print(f"📚 Datasets loaded, containing {len(datasets)} sub-datasets")

# Store all chunking results
new_datasets = {}

# Process all sub-datasets
print("\n📄 Starting to process all sub-datasets...")
for task_name in datasets:
    task_dataset = datasets[task_name]
    print(f"\n🔍 Processing sub-dataset: {task_name} (contains {len(task_dataset)} samples)")
    
    task_chunk_results = []
    
    for i, sample in enumerate(task_dataset):
        context = sample['context']
        query = sample['question']  # Use question field as query

        # Chunking - Monitor VRAM peak
        print(f"    📦 Chunking sample {i+1}/{len(task_dataset)}...")
        
        # Start time recording
        start_time = time.time()
        
        # Chunking
        chunks = chunker.chunk(context)
        
        # End time recording
        run_time = time.time() - start_time
        
        print(f"    📦 Generated {len(chunks)} chunks")
        
        # Create new sample, replace context with chunks, and add time field
        new_sample = {
            '_id': sample['_id'],
            'domain': sample['domain'],
            'sub_domain': sample['sub_domain'],
            'difficulty': sample['difficulty'],
            'length': sample['length'],
            'question': sample['question'],
            'choice_A': sample['choice_A'],
            'choice_B': sample['choice_B'],
            'choice_C': sample['choice_C'],
            'choice_D': sample['choice_D'],
            'answer': sample['answer'],
            'chunks': chunks,  # Replace original context field
            'time': run_time  # Run time
        }
        
        task_chunk_results.append(new_sample)

    # Create new dataset
    new_datasets[task_name] = Dataset.from_list(task_chunk_results)
    print(f"✅ {task_name} chunking completed, generated {len(task_chunk_results)} samples")

# Create new dataset dict
new_dataset_dict = DatasetDict(new_datasets)

# Save new datasets
print("\n💾 Saving new chunked datasets...")
new_dataset_dict.save_to_disk(f'./datasets_chunked/DenseX/{model_name}')
print(f"New chunked datasets saved to: ./datasets_chunked/DenseX/{model_name}")

print("\n✅ Processing completed!")

