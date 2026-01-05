from baseline.ppl_chunking import PPLChunking
from datasets import load_from_disk, Dataset, DatasetDict
import time
import os
from utils.monitor import Monitor
# Initialize monitor
monitor = Monitor(device_id="0")  # Use GPU 0

# Setup environment
monitor.setup()
model_name = 'Qwen2.5-1.5B-Instruct'
model_name_or_path = f'/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen2-1.5B-Instruct'
# Initialization
print("🚀 Initializing...")
chunker = PPLChunking(model_name_or_path=f'{model_name_or_path}', device='cuda')

root = './LongBench-v2'
datasets = load_from_disk(root)
print(f"📚 Datasets loaded, containing {len(datasets)} sub-datasets")
new_datasets = {}
print("\n📄 Starting to process all sub-datasets...")
for task_name in datasets:
    task_dataset = datasets[task_name]
    print(f"\n🔍 Processing sub-dataset: {task_name} (contains {len(task_dataset)} samples)")
    task_chunk_results = []
    for i, sample in enumerate(task_dataset):
        context = sample['context']
        query = sample['question']
        print(f"    📦 Chunking sample {i+1}/{len(task_dataset)}...")
        start_time = time.time()
        chunks = chunker.chunk(context, language='en')
        run_time = time.time() - start_time
        print(f"    📦 Generated {len(chunks)} chunks")
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
            'chunks': chunks,
            'time': run_time
        }
        task_chunk_results.append(new_sample)
    new_datasets[task_name] = Dataset.from_list(task_chunk_results)
    print(f"✅ {task_name} chunking completed, generated {len(task_chunk_results)} samples")
new_dataset_dict = DatasetDict(new_datasets)
save_base = './LongBench-v2_chunked/PPL'
os.makedirs(save_base, exist_ok=True)
save_path = os.path.join(save_base, model_name)
print("\n💾 Saving new chunked datasets...")
new_dataset_dict.save_to_disk(save_path)
print(f"New chunked datasets saved to: {save_path}")
print("\n✅ Processing completed!")
        
