# FreeChunker: A Cross-Granularity Chunking Framework

[English](README.md) | [中文](README_zh.md)

FreeChunker is a Cross-Granularity Encoding Framework that fundamentally transforms the traditional chunking paradigm for Retrieval-Augmented Generation (RAG). Unlike existing methods that rely on static boundary identification within fixed-granularity paradigms, FreeChunker treats sentences as atomic units and shifts from static chunk segmentation to flexible retrieval supporting arbitrary sentence combinations. This paradigm shift significantly avoids the computational overhead required for semantic boundary detection while enhancing adaptability to complex queries. Experimental evaluation on LongBench V2 demonstrates that FreeChunker possesses significant advantages in both retrieval performance and time efficiency compared to existing chunking methods.

## 🚀 Pre-trained Models

We provide pre-trained FreeChunker models initialized with state-of-the-art sentence embeddings on Hugging Face:

| Model | Base Model | Hugging Face Link |
|-------|------------|-------------------|
| **FreeChunk-Nomic** | `nomic-embed-text-v1.5` | [XiaSheng/FreeChunk-nomic](https://huggingface.co/XiaSheng/FreeChunk-nomic) |
| **FreeChunk-Jina** | `jina-embeddings-v2-small-en` | [XiaSheng/FreeChunk-jina](https://huggingface.co/XiaSheng/FreeChunk-jina) |
| **FreeChunk-BGE-M3** | `bge-m3` | [XiaSheng/FreeChunk-bge-m3](https://huggingface.co/XiaSheng/FreeChunk-bge-m3) |

## 📂 Repository Structure

This repository contains the source code for implementation, testing, and training of FreeChunker.

- **`src/`**: Core implementation of the FreeChunker framework.
  - `encoder.py`: The `UnifiedEncoder` class for end-to-end usage.
  - `freechunker.py`: The main model architecture.
  - `sentenizer.py`: Text splitting and backbone embedding integration.
  - `aggregator.py`: Post-retrieval text aggregation logic.
  
- **`test/`**: Unit tests and verification scripts.
  - Contains tests for different modules (`test_freechunker.py`, etc.) and baseline comparisons.

- **`train and preprocess/`**: Scripts for dataset construction and model training.
  - `1_build_pretrain_datasets.py`: Prepares training data.
  - `2_train_bge.py`, `3_train_jina.py`, `4_train_Nomic.py`: Training scripts for different backbones.
  - `*_chunk_*.py`: Chunking scripts for baselines (LumberChunker, Semantic Chunking, etc.).

- **`baseline/`**: Implementation of baseline chunking methods for comparison.

- **`upload_prep/`**: Prepared model files for Hugging Face deployment.

## 📦 Installation

```bash
pip install torch transformers sentence-transformers numpy
```

## ⚡ Quick Start

You can use the pre-trained models directly from Hugging Face using the `transformers` library.

```python
from transformers import AutoModel
import torch

# 1. Load Model (UnifiedEncoder)
# Replace with "XiaSheng/FreeChunk-jina" or "XiaSheng/FreeChunk-bge-m3" as needed
model = AutoModel.from_pretrained("XiaSheng/FreeChunk-nomic", trust_remote_code=True)

# 2. Build Vector Store from Text
text = "Your text..."
model.build_vector_store(text)

# 3. Query with Post-Aggregation (Default)
query = "Your query..."
results = model.query(query, top_k=1, aggregation_mode='post')

print(f"Query: {query}")
print(f"Result: {results}")
```

## 🛠 Training & Reproduction

To reproduce the training process or train on your own data:

1.  **Prepare Data**: Run `train and preprocess/1_build_pretrain_datasets.py` to generate the training corpus.
2.  **Train**: Execute the corresponding training script, e.g.:
    ```bash
    python "train and preprocess/4_train_Nomic.py"
    ```
3.  **Evaluate**: Use the scripts in `test/` to evaluate performance.

## 📄 Citation

If you use this code or models in your research, please cite:

```bibtex
@article{zhang2025freechunker, 
   title={FreeChunker: A Cross-Granularity Chunking Framework}, 
   author={Zhang, Wenxuan and Jiang, Yuan-Hao and Wu, Yonghe}, 
   journal={arXiv preprint arXiv:2510.20356}, 
   year={2025} 
 }
```

## 📝 License

This project is licensed under the Apache 2.0 License.
