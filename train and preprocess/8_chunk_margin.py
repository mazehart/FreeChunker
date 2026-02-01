from baseline.margin_sampling_chunking import MarginSamplingChunking
from datasets import load_dataset, Dataset, DatasetDict
import time
import os
import json
from tqdm import tqdm

# Setup environment
model_name = 'Qwen2.5-1.5B-Instruct'
model_name_or_path = f'Qwen/Qwen2-1.5B-Instruct'
# Initialization
print("🚀 Initializing...")
chunker = MarginSamplingChunking(model_name_or_path=model_name_or_path)

selected_domains = ['single-doc','multi-doc','code-repo','long-dialogue','long-icl','long-structured']
domains = {}
dataset_path = 'zai-org/LongBench-v2'
for dn in selected_domains:
    domains[dn] = load_dataset(dataset_path, dn)
print(f"📚 Datasets loaded, containing {len(domains)} domains")
new_datasets = {}
print("\n📄 Starting to process all sub-datasets...")
for dn, datasets in domains.items():
    if isinstance(datasets, DatasetDict):
        iterator = [(task_name, datasets[task_name]) for task_name in datasets]
    else:
        iterator = [("all", datasets)]
    task_chunk_results = []
    for task_name, task_dataset in iterator:
        print(f"\n🔍 Processing sub-dataset: {dn}/{task_name} (contains {len(task_dataset)} samples)")
        for i, sample in enumerate(tqdm(task_dataset, desc=f"📦 Chunking {dn}/{task_name}", unit="sample", leave=False)):
            context = sample['context']
            start_time = time.time()
            try:
                chunks = chunker.chunk(context, language='en')
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    chunks = []
                else:
                    raise
            run_time = time.time() - start_time
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
    new_datasets[dn] = Dataset.from_list(task_chunk_results)
    print(f"✅ {dn} chunking completed, generated {len(task_chunk_results)} samples")
save_base = './LongBench-v2_chunked/Margin'
new_dataset_root = os.path.join(save_base, model_name)
os.makedirs(new_dataset_root, exist_ok=True)
print("\n💾 Saving new chunked datasets...")
for dn, ds in new_datasets.items():
    out_dir = os.path.join(new_dataset_root, dn)
    ds.save_to_disk(out_dir)
index_path = os.path.join(new_dataset_root, "dataset_dict.json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump({"splits": list(new_datasets.keys())}, f, ensure_ascii=False)
print(f"New chunked datasets saved to: {new_dataset_root}")
print("\n✅ Processing completed!")
