from src.utils import generate_shifted_matrix
from datasets import load_from_disk, Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from tqdm import tqdm
from utils.monitor import Monitor
import os
import torch

# Initialize unified monitor
monitor = Monitor(device_id="2")
monitor.setup()
dataset_path = f'./corpus/val'
output_dir = f'./vector/jina-embeddings-v2-small-en/val'
model_name = 'jinaai/jina-embeddings-v2-small-en'
BATCH_SIZE = 8
SLICE_SIZE = 1000
START_IDX = 0      # Start index
END_IDX = 200    # End index, limit validation set size to 500

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

    # 组合句子 - 使用新的mask生成逻辑
    if n == 1:
        continue  # 单句子跳过，因为mask返回None
    
    matrix_result = generate_shifted_matrix(n)
    if matrix_result is None:
        continue  # 安全检查
    
    matrix = matrix_result[0]
    comb_list = []
    for col in range(matrix.shape[1]):
        indices = (matrix[:, col] == 1).nonzero(as_tuple=True)[0].tolist()
        if indices:
            combined = ' '.join([sentences[i] for i in indices])
            comb_list.append(combined)
    
    # 编码原始句子
    input_vectors = []
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i+BATCH_SIZE]
            vecs = model.encode(batch, show_progress_bar=False)
            input_vectors.extend(vecs)
            
        # 编码组合句子
        label_vectors = []
        for i in range(0, len(comb_list), BATCH_SIZE):
            batch = comb_list[i:i+BATCH_SIZE]
            vecs = model.encode(batch, show_progress_bar=False)
            label_vectors.extend(vecs)
        
    records.append({
        'input': input_vectors,
        'label': label_vectors
    })

    # 每1000条保存一次
    real_idx = START_IDX + idx
    if (idx + 1) % SLICE_SIZE == 0 or (idx + 1) == len(ds):
        part_ds = Dataset.from_list(records)
        part_path = f"{output_dir}/part_{real_idx // SLICE_SIZE}"
        part_ds.save_to_disk(part_path)
        part_paths.append(part_path)
        records = []
        part_idx += 1

print(f'数据处理完成，共生成 {len(part_paths)} 个分片文件:')
for part_path in part_paths:
    print(f'  - {part_path}')
