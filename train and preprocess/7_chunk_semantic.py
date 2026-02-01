from baseline.semantic_chunking import SemanticChunking
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import time
import os
import json

# Setup environment
embed_models = [
    'BAAI/bge-m3',
]
print("🚀 Initializing...")

selected_domains = ['single-doc','multi-doc','code-repo','long-dialogue','long-icl','long-structured']
domains = {}
dataset_path = 'zai-org/LongBench-v2'
for dn in selected_domains:
    domains[dn] = load_dataset(dataset_path, dn)
print(f"📚 Datasets loaded, containing {len(domains)} domains")

for model_path in embed_models:
    model_tag = os.path.basename(model_path)
    new_dataset_root = os.path.join('./LongBench-v2_chunked/Semantic', model_tag)
    checkpoints_root = os.path.join(new_dataset_root, "checkpoints")
    os.makedirs(checkpoints_root, exist_ok=True)
    
    chunker = SemanticChunking(embed_model_name=model_path, batch_size=16)
    new_datasets = {}
    print(f"\n Starting processing model: {model_tag}")
    
    # Calculate total samples for progress bar
    total_samples = 0
    for dn, datasets in domains.items():
        if isinstance(datasets, DatasetDict):
            total_samples += sum(len(datasets[task_name]) for task_name in datasets)
        else:
            total_samples += len(datasets)
    
    overall_pbar = tqdm(total=total_samples, desc=f"🌟 {model_tag} Overall Progress", unit="sample")
    
    for dn, datasets in domains.items():
        if isinstance(datasets, DatasetDict):
            iterator = [(task_name, datasets[task_name]) for task_name in datasets]
        else:
            iterator = [("all", datasets)]
        
        task_chunk_results = []
        checkpoint_path = os.path.join(checkpoints_root, f"{dn}.jsonl")
        
        # Read processed samples
        processed_samples = {}
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        _id = obj.get("_id")
                        if _id is not None:
                            processed_samples[_id] = obj
                    except:
                        pass
        
        for task_name, task_dataset in iterator:
            # print(f"\n🔍 Processing sub-dataset: {dn}/{task_name} (contains {len(task_dataset)} samples)")
            task_pbar = tqdm(task_dataset, desc=f"📦 {dn}/{task_name}", unit="sample", leave=False)
            
            for i, sample in enumerate(task_pbar):
                if sample['_id'] in processed_samples:
                    task_chunk_results.append(processed_samples[sample['_id']])
                    overall_pbar.update(1)
                    continue
                
                context = sample['context']
                start_time = time.time()
                chunks = chunker.chunk(context)
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
                
                # Save to checkpoint in real-time
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
                
                overall_pbar.update(1)
            task_pbar.close()
            
        new_datasets[dn] = Dataset.from_list(task_chunk_results)
        print(f"✅ {dn} chunking completed, generated {len(task_chunk_results)} samples")
    
    overall_pbar.close()
    
    for dn, ds in new_datasets.items():
        out_dir = os.path.join(new_dataset_root, dn)
        ds.save_to_disk(out_dir)
    print("\n💾 New chunked datasets saved")
    index_path = os.path.join(new_dataset_root, "dataset_dict.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"splits": list(new_datasets.keys())}, f, ensure_ascii=False)
    print(f"📇 Root directory index written: {index_path}")

print("\n✅ Processing completed!")
