from typing import List

def setup_tokenizer(model_name="/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B"):
    """Setup tokenizer"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def fixed_size_chunking(text: str, tokenizer=None, chunk_size: int = 256, overlap: int = 0) -> List[str]:
    """
    Fixed-size chunking based on token count (Strict truncation)
    
    Args:
        text: Text to chunk
        tokenizer: Tokenizer
        chunk_size: Token count per chunk
        overlap: Overlapping token count
    """
    if tokenizer is None:
        tokenizer = setup_tokenizer()
    
    # Encode the entire text, do not add special tokens to keep it clean
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    chunks = []
    
    # Calculate step size
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # Prevent infinite loop, theoretically overlap should be smaller than chunk_size
    
    for i in range(0, total_tokens, step):
        # Truncate tokens for current chunk
        chunk_tokens = tokens[i : i + chunk_size]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
            
    return chunks

def traditional_chunking(text, tokenizer=None, chunk_size=256, overlap=0):
    """
    Fixed-size chunking based on tokens
    
    Args:
        text: Text to chunk
        tokenizer: Tokenizer
        chunk_size: Token count per chunk
        overlap: Overlapping token count
    """
    return fixed_size_chunking(text, tokenizer, chunk_size, overlap)

class TraditionalChunking:
    def __init__(self, model_name_or_path=None, tokenizer=None, chunk_size=256, overlap=0):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name_or_path is not None:
            self.tokenizer = setup_tokenizer(model_name_or_path)
        else:
            self.tokenizer = setup_tokenizer()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        return traditional_chunking(text, self.tokenizer, self.chunk_size, self.overlap)
