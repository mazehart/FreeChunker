#!/usr/bin/env python3
"""
Sentenceizer - Universal sentence splitter + vector encoder
Length-constrained sentence splitting tool that protects special formats but not quotes/brackets
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from sentence_transformers import SentenceTransformer
from baseline.traditional_chunking import TraditionalChunking


class Sentenceizer:
    """
    Universal sentence splitter and encoder with length constraints, protecting special formats
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Sentenceizer
        
        Args:
            model_name (str, optional): SentenceTransformer model name
                                      If None, no encoding model is loaded
        """
        self.chunker = TraditionalChunking(chunk_size=256, overlap=0)
        
        self.model = None
        self.model_name = model_name
        if model_name:
            print(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            self.model.eval()
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def split(self, text: str) -> List[str]:
        """
        Split text into sentence list using NLTK sent_tokenize, then merge short sentences
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        if not text.strip():
            return []
        
        return self.chunker.chunk(text)
    
    def split_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text and return sentences with their positions in the original text
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, int, int]]: List of (sentence, start_position, end_position)
        """
        sentences = self.split(text)
        sentences_with_pos = []
        
        start_pos = 0
        for sentence in sentences:
            # Find sentence position in original text
            pos = text.find(sentence, start_pos)
            if pos != -1:
                sentences_with_pos.append((sentence, pos, pos + len(sentence)))
                start_pos = pos + len(sentence)
            else:
                # If not found (possibly due to merging or splitting), use estimated position
                sentences_with_pos.append((sentence, start_pos, start_pos + len(sentence)))
                start_pos += len(sentence)
        
        return sentences_with_pos
    
    def encode(self, text: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Encode text
        
        Args:
            text (Union[str, List[str]]): Input text, can be a single string or list of strings
                                         If it's a string, sentence splitting will be performed first
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Encoded vector array with shape (n_sentences, embedding_dim)
            
        Raises:
            ValueError: If no model is loaded
        """
        if self.model is None:
            raise ValueError("No model loaded. Please initialize with a model_name.")
        
        # If input is string, perform sentence splitting first
        if isinstance(text, str):
            sentences = self.split(text)
        else:
            sentences = text
        
        if not sentences:
            return np.array([])
        
        # Use sentence transformer for encoding, limit max batch size to 64
        embeddings = self.model.encode(
            sentences,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            batch_size=4
        )
        
        return embeddings
    
    def split_and_encode(self, text: str, show_progress: bool = True) -> Tuple[List[str], np.ndarray]:
        """
        Split text and encode
        
        Args:
            text (str): Input text
            show_progress (bool): Whether to show progress bar
            
        Returns:
            Tuple[List[str], np.ndarray]: (sentence list, encoded vector array)
        """
        sentences = self.split(text)
        embeddings = self.encode(sentences, show_progress=show_progress)
        return sentences, embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            raise ValueError("No model loaded.")
        return self.model.get_sentence_embedding_dimension()
    
def test_sentenceizer():
    """Test universal sentence splitting functionality and protection mechanisms"""
    
    print("=== Testing Universal Sentence Splitting and Protection Mechanisms ===")
    
    # Use reasonable length constraints for testing
    sentenceizer = Sentenceizer()
    
    test_cases = [
        # Basic sentence splitting test
        "This is the first sentence. This is the second sentence! This is the third sentence?",
        
        # Quote sentence splitting test (should be able to split)
        'He said "Hello there. How are you? I hope you are well." Then he left.',
        
        # Abbreviation protection test (should not split at abbreviations)
        "Dr. Smith is here. Mr. Jones left at 3 p.m. today. The U.S. economy is growing.",
        
        # Number protection test (should not split within numbers)
        "The temperature is 36.5 degrees. The price is $19.99. Version 2.1.3 was released.",
        
        # Ellipsis protection test (should not split at ellipsis)
        "This is incomplete... But this continues the thought. Another sentence follows.",
        
        # URL protection test (should not split within URLs)
        "Visit https://www.example.com for more info. The website www.test.org has details.",
        
        # Email protection test (should not split within emails)
        "Contact me at john.doe@example.com for questions. Send reports to admin@company.org please.",
        
        # Date and time protection test
        "The meeting is on 12/25/2023. We start at 3:30 p.m. today. See you then.",
        
        # Non-English text test
        "这是第一个句子。这是第二个句子！这是第三个句子？",
        
        # Mixed text test
        "This is English. 这是中文。Mix of both languages!",
        
        # Complex mixed test
        "访问 https://www.baidu.com 获取信息。联系邮箱是 test@163.com。价格为 ￥99.99 元。",
        
        # Long sentence test (should be split)
        "This is a very long sentence that should be split into multiple parts because it exceeds the maximum length limit that we have set for individual sentences in our system, and we need to handle this properly.",
        
        # Sentences starting with numbers
        "Today is sunny. 123 people attended the meeting. Everyone was happy.",
        
        # Sentences starting with special characters
        "First sentence here. \"Quoted sentence comes next.\" Final sentence ends it.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Original: {text}")
        
        sentences = sentenceizer.split(text)
        print(f"Split Result ({len(sentences)} sentences):")
        for j, sentence in enumerate(sentences, 1):
            print(f"  {j}. ({len(sentence)} chars) {sentence}")
        

if __name__ == "__main__":
    test_sentenceizer() 