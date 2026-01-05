#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
modfied from lmchunker.modules.dense_x_retrieval
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lmchunker.modules.dense_x_retrieval import dense_x_retrieval as dx_retrieval

def setup_propositionizer_model(model_name="chentong00/propositionizer-wiki-flan-t5-large", device="cuda"):
    """设置命题生成模型"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return model, tokenizer

def dense_x_retrieval_chunking(text, model=None, tokenizer=None):
    
    if model is None or tokenizer is None:
        model, tokenizer = setup_propositionizer_model()
    
    chunks = dx_retrieval(
        tokenizer=tokenizer,
        model=model,
        text=text,
        title='',
        section='',
        target_size=256
    )
    return chunks

class DenseXRetrievalChunking:
    def __init__(self, model_name="chentong00/propositionizer-wiki-flan-t5-large", device="cuda"):
        self.model, self.tokenizer = setup_propositionizer_model(model_name, device)

    def chunk(self, text):
        return dense_x_retrieval_chunking(text, self.model, self.tokenizer) 

if __name__ == "__main__":
    setup_propositionizer_model(device="mps")