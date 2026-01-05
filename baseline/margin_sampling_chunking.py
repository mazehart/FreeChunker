import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lmchunker.modules.margin_sampling_chunking import llm_chunker_ms

def setup_model(
    model_name_or_path='Qwen2-1.5B-Instruct',
    device="cuda",
    torch_dtype="bfloat16",
    use_cache=False,
    attn_implementation="sdpa"
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    dtype = None
    if torch.cuda.is_available():
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=dtype
    )
    model.config.use_cache = use_cache
    model.config.attn_implementation = attn_implementation
    model = model.to(device)
    model.eval()
    return model, tokenizer

def margin_sampling_chunking(
    sub_text,
    model=None,
    tokenizer=None,
    language='en',
    dynamic_merge='no',
    target_size=256
):
    if model is None or tokenizer is None:
        model, tokenizer = setup_model()
    with torch.inference_mode():
        try:
            return llm_chunker_ms(sub_text, model, tokenizer, language, dynamic_merge, target_size)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
    chunks_acc = []
    ids = tokenizer.encode(sub_text, add_special_tokens=False)
    WINDOW_TOKENS = 4096
    for start in range(0, len(ids), WINDOW_TOKENS):
        part_ids = ids[start:start + WINDOW_TOKENS]
        part_text = tokenizer.decode(part_ids)
        with torch.inference_mode():
            try:
                partial = llm_chunker_ms(part_text, model, tokenizer, language, dynamic_merge, target_size)
                if isinstance(partial, list):
                    chunks_acc.extend(partial)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    break
                else:
                    raise
    return chunks_acc

class MarginSamplingChunking:
    def __init__(
        self,
        model_name_or_path='Qwen2-1.5B-Instruct',
        device="cuda"
    ):
        self.model, self.tokenizer = setup_model(
            model_name_or_path=model_name_or_path,
            device=device,
        )

    def chunk(self, sub_text, language='en'):
        return margin_sampling_chunking(
            sub_text,
            self.model,
            self.tokenizer,
            language
        )
