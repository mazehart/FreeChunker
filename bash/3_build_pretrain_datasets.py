from src.utils import generate_shifted_matrix
from datasets import load_from_disk, Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from tqdm import tqdm
from utils.monitor import Monitor
import os
import torch

# Initialize unified monitor
monitor = Monitor(device_id="5")
monitor.setup()
dataset_path = f'./corpus/train'
output_dir = f'./vector/jina-embeddings-v2-small-en'
model_name = '/share/home/ecnuzwx/UnifiedRAG/cache/models--jinaai--jina-embeddings-v2-small-en'
BATCH_SIZE = 8
SLICE_SIZE = 1000
START_IDX = 50000     # Start index
END_IDX = 75000   # End index, set to None to process until the end

# Check if output directory exists, create if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Loading model...')
model = SentenceTransformer(model_name, trust_remote_code=True, device='cuda')
model.eval()  # Set to inference mode
print('Model loaded')

print('Loading dataset...')
ds = load_from_disk(dataset_path)
print(f'Dataset loaded, total {len(ds)} records')

# Select data range
end_idx = END_IDX if END_IDX is not None else len(ds)
end_idx = min(end_idx, len(ds))  # Ensure not exceeding dataset size

if START_IDX > 0 or end_idx < len(ds):
    ds = ds.select(range(START_IDX, end_idx))
    print(f'Selected data range [{START_IDX}:{end_idx}], total {len(ds)} records')

records = []
part_idx = 0
part_paths = []

for idx in tqdm(range(len(ds)), desc='Encoding'):
    item = ds[idx]
    sentences = item['sentences']
    n = len(sentences)

    # Combine sentences - using new mask generation logic
    if n == 1:
        continue  # Skip single sentence because mask returns None
    
    matrix_result = generate_shifted_matrix(n)
    if matrix_result is None:
        continue  # Safety check
    
    matrix = matrix_result[0]
    comb_list = []
    for col in range(matrix.shape[1]):
        indices = (matrix[:, col] == 1).nonzero(as_tuple=True)[0].tolist()
        if indices:
            combined = ' '.join([sentences[i] for i in indices])
            comb_list.append(combined)
    
    # Encode original sentences
    input_vectors = []
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i+BATCH_SIZE]
            vecs = model.encode(batch, show_progress_bar=False)
            input_vectors.extend(vecs)
            
        # Encode combined sentences
        label_vectors = []
        for i in range(0, len(comb_list), BATCH_SIZE):
            batch = comb_list[i:i+BATCH_SIZE]
            vecs = model.encode(batch, show_progress_bar=False)
            label_vectors.extend(vecs)
        
    records.append({
        'input': input_vectors,
        'label': label_vectors
    })

    # Save every 1000 records
    real_idx = START_IDX + idx
    if (idx + 1) % SLICE_SIZE == 0 or (idx + 1) == len(ds):
        part_ds = Dataset.from_list(records)
        part_path = f"{output_dir}/part_{real_idx // SLICE_SIZE}"
        part_ds.save_to_disk(part_path)
        part_paths.append(part_path)
        records = []
        part_idx += 1

print(f'Data processing completed, generated {len(part_paths)} part files:')
for part_path in part_paths:
    print(f'  - {part_path}')