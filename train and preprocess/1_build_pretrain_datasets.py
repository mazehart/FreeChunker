import os
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils import generate_shifted_matrix

# Configuration
TASKS = [
    {
        "name": "train",
        "dataset_path": "XiaSheng/FreeChunk-corpus",
        "split": "train",
        "output_dir": "./vector/jina-embeddings-v2-small-en",
        "model_name": "jinaai/jina-embeddings-v2-small-en",
        "batch_size": 8,
        "slice_size": 1000,
        "start_idx": 0,
        "end_idx": None  # Process all
    },
    {
        "name": "val",
        "dataset_path": "XiaSheng/FreeChunk-corpus",
        "split": "validation",
        "output_dir": "./vector/jina-embeddings-v2-small-en/val",
        "model_name": "jinaai/jina-embeddings-v2-small-en",
        "batch_size": 8,
        "slice_size": 1000,
        "start_idx": 0,
        "end_idx": 200
    },
    {
        "name": "test",
        "dataset_path": "XiaSheng/FreeChunk-corpus",
        "split": "validation",
        "output_dir": "./vector/jina-embeddings-v2-small-en/test",
        "model_name": "jinaai/jina-embeddings-v2-small-en",
        "batch_size": 8,
        "slice_size": 1000,
        "start_idx": 200,
        "end_idx": 500
    }
]

def process_task(task):
    print(f"\n===== Starting Task: {task['name']} =====")
    
    # Unpack config
    dataset_path = task['dataset_path']
    output_dir = task['output_dir']
    model_name = task['model_name']
    batch_size = task.get('batch_size', 8)
    slice_size = task.get('slice_size', 1000)
    start_idx = task.get('start_idx', 0)
    end_idx = task.get('end_idx')

    # Ensure output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load Model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device='cuda')
    model.eval()
    print("Model loaded")

    # Load Dataset
    print(f"Loading dataset: {dataset_path}")
    split = task.get('split', 'train')
    ds = load_dataset(dataset_path, split=split)
    
    print(f"Dataset loaded, total {len(ds)} records")

    # Slice Dataset
    total_len = len(ds)
    effective_end = end_idx if end_idx is not None else total_len
    effective_end = min(effective_end, total_len)
    
    if start_idx > 0 or effective_end < total_len:
        ds = ds.select(range(start_idx, effective_end))
        print(f"Selected range [{start_idx}:{effective_end}], processing {len(ds)} records")
    else:
        print(f"Processing all {len(ds)} records")

    # Processing Loop
    records = []
    part_idx = 0
    part_paths = []

    for i in tqdm(range(len(ds)), desc=f"Encoding ({task['name']})"):
        item = ds[i]
        sentences = item['sentences']
        n = len(sentences)

        if n == 1:
            continue

        matrix_result = generate_shifted_matrix(n)
        if matrix_result is None:
            continue
        
        matrix = matrix_result[0]
        comb_list = []
        for col in range(matrix.shape[1]):
            indices = (matrix[:, col] == 1).nonzero(as_tuple=True)[0].tolist()
            if indices:
                combined = ' '.join([sentences[idx] for idx in indices])
                comb_list.append(combined)
        
        # Encode
        input_vectors = []
        label_vectors = []
        
        with torch.no_grad():
            # Original sentences
            for j in range(0, len(sentences), batch_size):
                batch = sentences[j:j+batch_size]
                vecs = model.encode(batch, show_progress_bar=False)
                input_vectors.extend(vecs)
            
            # Combined sentences
            for j in range(0, len(comb_list), batch_size):
                batch = comb_list[j:j+batch_size]
                vecs = model.encode(batch, show_progress_bar=False)
                label_vectors.extend(vecs)

        records.append({
            'input': input_vectors,
            'label': label_vectors
        })

        # Save slice
        real_idx = start_idx + i
        if (i + 1) % slice_size == 0 or (i + 1) == len(ds):
            part_ds = Dataset.from_list(records)
            part_path = os.path.join(output_dir, f"part_{real_idx // slice_size}")
            part_ds.save_to_disk(part_path)
            part_paths.append(part_path)
            records = []
            part_idx += 1
    
    print(f"Task {task['name']} completed. Generated {len(part_paths)} part files.")

def main():
    for task in TASKS:
        process_task(task)

if __name__ == "__main__":
    main()
