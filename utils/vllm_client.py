from openai import OpenAI
from transformers import AutoTokenizer


class VLLMClient:
    def __init__(self, system_prompt="You are an excellent reading comprehension assistant. Please provide answers in JSON format."):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8888/v1"
        )
        self.system_prompt = system_prompt
        self.device = "vllm-server"
        self.tokenizer = AutoTokenizer.from_pretrained("/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B")
        self.max_context_length = 40000
        self.reserved_tokens = 1000

    def chat(self, text, **kwargs):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        request_params = {
            "model": "/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0,
            "extra_body": {
                "do_sample": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content.strip()