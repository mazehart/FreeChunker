from baseline.lumberchunker import LumberChunking
from datasets import load_dataset, Dataset, DatasetDict
from openai import OpenAI
from tqdm import tqdm
import time
import os
import json
from transformers import AutoTokenizer

model_name = 'qwen3-8b'
model_name_or_path = "Qwen/Qwen3-8B"

# Initialization

class VLLMClient:
    def __init__(self, system_prompt="You are an excellent reading comprehension assistant. Please provide answers in JSON format."):
        self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8888/v1")
        self.system_prompt = system_prompt
        self.model_path = model_name_or_path
        print(f"✅ vLLM client initialization completed")
        print(f"🎲 Generation parameters: Deterministic output mode")
        print("🔧 Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_context_length = 40000
        self.reserved_tokens = 1000
        print(f"📏 Max context length: {self.max_context_length}, reserved tokens: {self.reserved_tokens}")
    
    def chat(self, text, **kwargs):
        """Generate answer"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        request_params = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0,
            "extra_body": {
                "do_sample": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content.strip()
    

# Initialize vLLM API client
llm = VLLMClient(system_prompt="You are an AI human assistant, answering user questions as perfectly as possible.")
chunker = LumberChunking(llm=llm)

print("🚀 Initializing...")

model_tag = model_name
new_dataset_root = os.path.join('./LongBench-v2_chunked/Lumber', model_tag)
checkpoints_root = os.path.join(new_dataset_root, "checkpoints")
os.makedirs(checkpoints_root, exist_ok=True)

selected_domains = ['single-doc','multi-doc']
domains = {}
dataset_path = 'zai-org/LongBench-v2'
for dn in selected_domains:
    domains[dn] = load_dataset(dataset_path, dn)

print(f"📚 Datasets loaded, containing {len(domains)} domains")

new_datasets = {}

total_samples = 0
for dn, datasets in domains.items():
    if isinstance(datasets, DatasetDict):
        total_samples += sum(len(datasets[task_name]) for task_name in datasets)
    else:
        total_samples += len(datasets)

overall_pbar = tqdm(total=total_samples, desc="🌟 Overall Progress", unit="sample")

for dn, datasets in domains.items():
    print("\n📄 Starting to process all sub-datasets...")
    if isinstance(datasets, DatasetDict):
        iterator = [(task_name, datasets[task_name]) for task_name in datasets]
    else:
        iterator = [("all", datasets)]
    task_chunk_results = []
    checkpoint_path = os.path.join(checkpoints_root, f"{dn}.jsonl")
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
        print(f"\n🔍 Processing sub-dataset: {dn}/{task_name} (contains {len(task_dataset)} samples)")
        task_pbar = tqdm(task_dataset, desc=f"📦 {dn}/{task_name}", unit="sample", leave=False)
        for i, sample in enumerate(task_pbar):
            if sample['_id'] in processed_samples:
                task_chunk_results.append(processed_samples[sample['_id']])
                overall_pbar.update(1)
                continue
            context = sample['context']
            start_time = time.time()
            chunks = chunker.chunk(context, timeout_seconds=600)
            run_time = time.time() - start_time
            task_pbar.set_postfix({
                "Chunks": len(chunks) if chunks is not None else 0,
                "Time": f"{run_time:.1f}s"
            })
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
                'chunks': chunks if chunks is not None else [],
                'time': run_time
            }
            task_chunk_results.append(new_sample)
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
