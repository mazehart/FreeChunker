from baseline.traditional_chunking import TraditionalChunking
from datasets import load_dataset, Dataset, DatasetDict
import time
import os
import json

model_name = 'Qwen/Qwen3-8B'
model_name_or_path = model_name
print("🚀 Initializing...")

# All domain names and selective processing list
all_domains = ['single-doc','multi-doc','code-repo','long-dialogue','long-icl','long-structured']
selected_domains = ['single-doc','multi-doc','code-repo','long-dialogue','long-icl','long-structured']

# Always load selected domains by subdirectory
domains = {}
dataset_path = 'zai-org/LongBench-v2'
for dn in selected_domains:
    domains[dn] = load_dataset(dataset_path, dn)
print(f"📚 Datasets loaded, containing {len(domains)} domains")

size = 256
chunker = TraditionalChunking(model_name_or_path=f'{model_name_or_path}', tokenizer=None, chunk_size=size, overlap=0)
for dn, datasets in domains.items():
    sub_count = len(datasets) if isinstance(datasets, DatasetDict) else 1
    print(f"\n📄 Processing domain: {dn} (contains {sub_count} sub-datasets)")
    merged_results = []
    if isinstance(datasets, DatasetDict):
        iterator = [(task_name, datasets[task_name]) for task_name in datasets]
    else:
        iterator = [("all", datasets)]
    for task_name, task_dataset in iterator:
        print(f"\n🔍 Processing sub-dataset: {task_name} (contains {len(task_dataset)} samples)")
        task_chunk_results = []
        for i, sample in enumerate(task_dataset):
            context = sample['context']
            query = sample['question']
            print(f"    📦 Chunking sample {i+1}/{len(task_dataset)}...")
            start_time = time.time()
            chunks = chunker.chunk(context)
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
        merged_results.extend(task_chunk_results)
        print(f"✅ {task_name} chunking completed, generated {len(task_chunk_results)} samples, merged into list")
    merged_dataset = Dataset.from_list(merged_results)
    out_dir = f'./LongBench-v2_chunked/Traditional/{size}/{dn}'
    print("\n💾 Saving new chunked datasets...")
    merged_dataset.save_to_disk(out_dir)
    print(f"New chunked datasets saved to: {out_dir}")
root_out = f'./LongBench-v2_chunked/Traditional/{size}'
os.makedirs(root_out, exist_ok=True)
index_path = os.path.join(root_out, "dataset_dict.json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump({"splits": list(domains.keys())}, f, ensure_ascii=False)
print(f"📇 Root directory index written: {index_path}")
