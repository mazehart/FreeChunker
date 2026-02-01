#!/usr/bin/env python3
"""
UnifiedEncoder - Unified text encoder
Integrates sentence splitting and multiple encoding models into a unified interface
"""

import torch
import numpy as np
import pickle
import os
from typing import List, Tuple, Union
from src.sentenizer import Sentenceizer
from src.freechunker import FreeChunkerModel
from src.aggregator import TextAggregator

class UnifiedEncoder:
    """
    Unified text encoder, supporting text sentence splitting and encoding for multiple models
    """
    
    def __init__(self, model_name: str, local_model_path: str = None, granularities: List[int] = None):
        """
        Initialize unified text encoder
        
        Args:
            model_name (str): Model name
            local_model_path (str, optional): Local model path for loading fine-tuned weights
            granularities (List[int], optional): Granularities for chunking
        """
        self.model_name = model_name
        self.granularities = granularities
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        
        # Initialize text aggregator
        self.aggregator = TextAggregator()
        
        print(f"Initializing unified text encoder, model: {model_name}")
        print(f"Using local model path: {local_model_path}")
        print(f"Using device: {self.device}")

        self.model = FreeChunkerModel.from_pretrained(local_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Select model and preprocessor based on model name
        # Predefined model mapping: name -> HF_model_ID
        model_configs = {
            'bge-m3': 'BAAI/bge-m3',
            'nomic-embed-text-v1.5': 'nomic-ai/nomic-embed-text-v1.5',
            'jina': 'jinaai/jina-embeddings-v2-small-en'
        }

        if model_name in model_configs:
            hf_id = model_configs[model_name]
            self.sentenceizer = Sentenceizer(model_name=hf_id)
        else:
            # Try using model_name directly as path or ID
            print(f"Unknown predefined model name: {model_name}, trying to load directly...")
            self.sentenceizer = Sentenceizer(model_name=model_name)
            
        print("Unified text encoder initialized!")

    def encode(self, text: str, show_progress: bool = True) -> Tuple[List[str], np.ndarray, List[List[str]]]:
        """
        Split text and encode, return results grouped by shift_matrix
        
        Args:
            text (str): Input text
            show_progress (bool): Whether to show progress
            
        Returns:
            Tuple[List[str], np.ndarray, List[List[str]]]: (Original sentence list, encoded vector array, grouped sentence list by shift_matrix)
        """
        with torch.no_grad():
            sentences, input_embeddings = self.sentenceizer.split_and_encode(text, show_progress=show_progress)
            
            if len(sentences) == 0:
                return sentences, np.array([]), []
            if isinstance(input_embeddings, np.ndarray):
                input_embeddings = torch.from_numpy(input_embeddings)
            input_embeddings = input_embeddings.to(self.device)
            inputs_embeds = input_embeddings.unsqueeze(0)
            outputs = self.model(inputs_embeds=inputs_embeds, granularities=self.granularities)
            final_embeddings = outputs['embedding']
            shift_matrix = outputs['shift_matrix']
            
            # Group sentences using shift_matrix
            sentences = [f"【Begin-{num}】" + sentence + f"【End-{num}】" for num, sentence in enumerate(sentences)]
            grouped_sentences = self._group_sentences_by_shift_matrix(sentences, shift_matrix)
            result_embeddings = final_embeddings.cpu().numpy()
            
            return sentences, result_embeddings, grouped_sentences
    
    def _group_sentences_by_shift_matrix(self, sentences: List[str], shift_matrix: torch.Tensor) -> List[List[str]]:
        """
        Group sentences according to shift_matrix (Optimized version)
        
        Args:
            sentences (List[str]): Original sentence list
            shift_matrix (torch.Tensor): Mask matrix with shape [num_chunks, seq_len]
            
        Returns:
            List[List[str]]: List of sentences grouped by shift_matrix
        """
        
        grouped_sentences = []
        num_chunks, seq_len = shift_matrix.shape
        
        for chunk_idx in range(num_chunks):
            chunk_mask = shift_matrix[chunk_idx]  # [seq_len]
            
            # Use vectorized operation to get all indices that are 1
            valid_indices = (chunk_mask == 1).nonzero(as_tuple=True)[0].cpu().numpy()
            
            # Select only indices within the sentence list range
            valid_indices = valid_indices[valid_indices < len(sentences)]
            
            if len(valid_indices) > 0:
                # Get sentences directly by index
                chunk_sentences = [sentences[idx] for idx in valid_indices]
                grouped_sentences.append(chunk_sentences)
                
        return grouped_sentences

    def build_vector_store(self, text: str, show_progress: bool = True):
        """
        Build vector store based on long text
        
        Args:
            text (str): Long text
            show_progress (bool): Whether to show progress
        """
        
        sentences, embeddings, grouped_sentences = self.encode(text, show_progress)
        
        # grouped_texts = [" ".join(group) if isinstance(group, list) else str(group) for group in grouped_sentences]

        grouped_texts = sentences + [" ".join(group) if isinstance(group, list) else str(group) for group in grouped_sentences]
        
        self.vector_store = {
            'sentences': sentences,  # Keep original sentences for debugging
            'embeddings': embeddings,  # embeddings correspond to grouped_sentences
            'grouped_sentences': grouped_sentences,  # Original grouping structure
            'grouped_texts': grouped_texts  # Text for retrieval
        }
        
        if show_progress:
            print(f"Vector store built: {len(sentences)} original sentences, {len(grouped_sentences)} groups, {len(embeddings)} embedding vectors")
            print(f"Vector store verification: embeddings.shape={embeddings.shape}, grouped_texts count={len(grouped_texts)}\n")
    
    def query(self, query: str, top_k: int = 5, aggregation_mode: str = 'post', tokenizer=None) -> Union[List[Tuple[str, float]], str]:
        """
        Query vector store
        
        Args:
            query (str): Query text
            top_k (int): Return top k most similar results
            aggregation_mode (str): Aggregation mode
                - 'none': No aggregation, return top_k results directly [(text, score), ...]
                - 'post': Post-aggregation mode, return aggregated text string
            
        Returns:
            Union[List[Tuple[str, float]], str]: 
                - If aggregation_mode='none', return [(sentence, similarity_score), ...]
                - If aggregation_mode='post', return aggregated string
        """
        if not hasattr(self, 'vector_store'):
            raise ValueError("Vector store not built, please call build_vector_store method first")
        
        # Encode query text
        query_embeddings = self.sentenceizer.encode([query])
        query_embedding = query_embeddings[0]

        # Calculate cosine similarity
        similarities = np.dot(self.vector_store['embeddings'], query_embedding)
        
        # Sort (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        if aggregation_mode == 'none':
            return self._get_direct_results(sorted_indices, similarities, top_k)
        elif aggregation_mode == 'post':
            return self._post_aggregation(sorted_indices, similarities, top_k, tokenizer=tokenizer)
        else:
            print(f"Warning: Unknown aggregation_mode '{aggregation_mode}', falling back to 'none'")
            return self._get_direct_results(sorted_indices, similarities, top_k)
            
    def _get_direct_results(self, sorted_indices: np.ndarray, similarities: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        
        available_count = len(self.vector_store['grouped_texts'])
        actual_top_k = min(top_k, available_count)
        top_indices = sorted_indices[:actual_top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.vector_store['grouped_texts']):
                grouped_text = self.vector_store['grouped_texts'][idx]
                score = similarities[idx]
                results.append((grouped_text, float(score)))
        
        return results
    
    def _post_aggregation(self, sorted_indices: np.ndarray, similarities: np.ndarray, top_k: int, tokenizer=None) -> List[Tuple[str, float]]:
        
        # Get top_k results first
        direct_results = self._get_direct_results(sorted_indices, similarities, top_k)
        
        # Extract text parts for aggregation
        texts = [text for text, score in direct_results]
        
        aggregated_texts = self.aggregator.aggregate_segments(texts)
        
        
        return aggregated_texts
        
    
    def load_vector_store(self, file_path: str):
        """
        Load vector store from file
        
        Args:
            file_path (str): Vector store file path
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vector store file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            self.vector_store = pickle.load(f)
        
        print(f"Vector store loaded from {file_path}")
        print(f"Vector store info: {len(self.vector_store['grouped_texts'])} groups, embedding dimension: {self.vector_store['embeddings'].shape}")
    
    def has_vector_store(self) -> bool:
        """
        Check if vector store is built or loaded
        
        Returns:
            bool: Whether a vector store is available
        """
        return hasattr(self, 'vector_store') and self.vector_store is not None
