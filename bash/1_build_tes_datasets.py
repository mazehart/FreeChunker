from src.utils import generate_shifted_matrix
from datasets import load_from_disk, Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from tqdm import tqdm
from utils.monitor import Monitor
import os
import torch

# 初始化统一监视器
monitor = Monitor(device_id="1")
monitor.setup()
dataset_path = f'./corpus/val'
output_dir = f'./vector/nomic-embed-text-v1.5/test'
model_name = 'nomic-ai/nomic-embed-text-v1.5'
BATCH_SIZE = 8
SLICE_SIZE = 1000
START_IDX = 200   
END_IDX = 500    

# 检查输出文件夹是否存在，不存在则新建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('开始加载模型...')
model = SentenceTransformer(model_name, trust_remote_code=True, device='cuda')
model.eval()  # 设置为推理模式
print('模型加载完成')


print('开始加载数据集...')
ds = load_from_disk(dataset_path)
print(f'数据集加载完成，共有 {len(ds)} 条数据')

# 选择数据范围
end_idx = END_IDX if END_IDX is not None else len(ds)
end_idx = min(end_idx, len(ds))  # 确保不超过数据集大小

if START_IDX > 0 or end_idx < len(ds):
    ds = ds.select(range(START_IDX, end_idx))
    print(f'选择数据范围 [{START_IDX}:{end_idx}]，共 {len(ds)} 条数据')

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
