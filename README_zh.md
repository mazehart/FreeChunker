# FreeChunker: 跨粒度分块框架

[English](README.md) | [中文](README_zh.md)

FreeChunker 是一个跨粒度编码框架，彻底改变了检索增强生成 (RAG) 的传统分块范式。现有的方法通常依赖于固定粒度范式下的静态边界识别，限制了对多样化查询需求的适应性。FreeChunker 将句子视为原子单元，从静态分块分割转变为支持任意句子组合的灵活检索。这种范式转变不仅显著避免了语义边界检测所需的计算开销，还增强了对复杂查询的适应性。LongBench V2 上的实验评估表明，与现有的分块方法相比，FreeChunker 在检索性能和时间效率方面都具有显著优势。

## 🚀 预训练模型

我们在 Hugging Face 上提供了使用最先进的句子 Embedding 初始化的预训练 FreeChunker 模型：

| 模型 | 基础模型 (Base Model) | Hugging Face 链接 |
|-------|------------|-------------------|
| **FreeChunk-Nomic** | `nomic-embed-text-v1.5` | [XiaSheng/FreeChunk-nomic](https://huggingface.co/XiaSheng/FreeChunk-nomic) |
| **FreeChunk-Jina** | `jina-embeddings-v2-small-en` | [XiaSheng/FreeChunk-jina](https://huggingface.co/XiaSheng/FreeChunk-jina) |
| **FreeChunk-BGE-M3** | `bge-m3` | [XiaSheng/FreeChunk-bge-m3](https://huggingface.co/XiaSheng/FreeChunk-bge-m3) |

## 📂 仓库结构

本仓库包含 FreeChunker 的实现、测试和训练源代码。

- **`src/`**: FreeChunker 框架的核心实现。
  - `encoder.py`: 用于端到端使用的 `UnifiedEncoder` 类。
  - `freechunker.py`: 主模型架构。
  - `sentenizer.py`: 文本分割和骨干 Embedding 集成。
  - `aggregator.py`: 检索后的文本聚合逻辑。
  
- **`test/`**: 单元测试和验证脚本。
  - 包含不同模块的测试 (`test_freechunker.py` 等) 和基线对比。

- **`train and preprocess/`**: 数据集构建和模型训练脚本。
  - `1_build_pretrain_datasets.py`: 准备训练数据。
  - `2_train_bge.py`, `3_train_jina.py`, `4_train_Nomic.py`: 不同骨干模型的训练脚本。
  - `*_chunk_*.py`: 基线分块方法的脚本 (LumberChunker, Semantic Chunking 等)。

- **`baseline/`**: 用于比较的基线分块方法实现。

- **`upload_prep/`**: 准备好用于部署到 Hugging Face 的模型文件。

## 📦 安装

```bash
pip install torch transformers sentence-transformers numpy
```

## ⚡ 快速开始

您可以直接使用 `transformers` 库从 Hugging Face 加载预训练模型。

```python
from transformers import AutoModel
import torch

# 1. 加载模型 (UnifiedEncoder)
# 根据需要替换为 "XiaSheng/FreeChunk-jina" 或 "XiaSheng/FreeChunk-bge-m3"
model = AutoModel.from_pretrained("XiaSheng/FreeChunk-nomic", trust_remote_code=True)

# 2. 从文本构建向量库
text = "Your text..."
model.build_vector_store(text)

# 3. 使用后聚合 (Post-Aggregation) 进行查询 (默认)
query = "Your query..."
results = model.query(query, top_k=1, aggregation_mode='post')

print(f"Query: {query}")
print(f"Result: {results}")
```

## 🛠 训练与复现

要复现训练过程或在您自己的数据上进行训练：

1.  **准备数据**: 运行 `train and preprocess/1_build_pretrain_datasets.py` 生成训练语料。
2.  **训练**: 执行相应的训练脚本，例如：
    ```bash
    python "train and preprocess/4_train_Nomic.py"
    ```
3.  **评估**: 使用 `test/` 中的脚本来评估性能。

## 📄 引用

如果您在研究中使用了此代码或模型，请引用：

```bibtex
@article{zhang2025freechunker, 
   title={FreeChunker: A Cross-Granularity Chunking Framework}, 
   author={Zhang, Wenxuan and Jiang, Yuan-Hao and Wu, Yonghe}, 
   journal={arXiv preprint arXiv:2510.20356}, 
   year={2025} 
 }
```

## 📝 许可证

本项目采用 Apache 2.0 许可证。
