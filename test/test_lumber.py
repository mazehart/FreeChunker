
import torch
import numpy as np
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from datetime import datetime
import json_repair
from openai import OpenAI
import requests
import time
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import io
import sys
import statistics

class EmbeddingModel:
    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer
        print(f"🔧 Loading embedding model: {model_path}")

        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  
        
        print(f"✅ Embedding model loaded, device: {self.device}")
        print(f"📏 Model max sequence length: {self.model.max_seq_length}")
        print(f"🎯 Model dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts, batch_size=6):
        """Encode text"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  
        )
        
        return embeddings

class VLLMClient:
    def __init__(self, system_prompt="You are an excellent reading comprehension assistant. Please provide answers in JSON format."):
        """Connect directly to vLLM using OpenAI client"""
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8888/v1"
        )
        self.system_prompt = system_prompt
        self.device = "vllm-server"

        print("🔧 Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.max_context_length = 40000  
        self.reserved_tokens = 1000  
        
        print(f"✅ vLLM client initialized")
        print(f"🎲 Generation params: Deterministic mode")
        print(f"📏 Max context length: {self.max_context_length}, Reserved tokens: {self.reserved_tokens}")
    
    def chat(self, text, **kwargs):
        """Generate answer"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        
        request_params = {
            "model": "Qwen/Qwen3-8B",
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
    
    def truncate_retrieved_context(self, retrieved_context, question):
        """Truncate retrieved context to ensure it does not exceed max context length"""
        
        system_tokens = len(self.tokenizer.encode(self.system_prompt))
        question_tokens = len(self.tokenizer.encode(question))

        max_context_tokens = self.max_context_length - system_tokens - question_tokens - self.reserved_tokens

        context_tokens = self.tokenizer.encode(retrieved_context)
        if len(context_tokens) > max_context_tokens:
            print(f"⚠️  Retrieved context too long ({len(context_tokens)} tokens), truncated to {max_context_tokens} tokens")
            truncated_tokens = context_tokens[:max_context_tokens]
            retrieved_context = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return retrieved_context

class QASystem:
    def __init__(self, 
                 lumber_dataset_path='./datasets_chunked/Lumber',
                 embedding_model_path='jinaai/jina-embeddings-v2-small-en'):
        
        print("🚀 Initializing Lumber QA System...")

        print("📊 Loading Lumber dataset...")
        self.datasets = load_from_disk(lumber_dataset_path)
        print(f"✅ Dataset loaded, domains: {list(self.datasets.keys())}")

        self.embedding_model = EmbeddingModel(embedding_model_path)

        print(f"🔧 Initializing LLM (vLLM API mode)...")

        if not self._check_vllm_server():
            raise RuntimeError("❌ vLLM server unavailable! Please start service provided by server.py")
        
        try:
            self.llm = VLLMClient()
            print("✅ vLLM API client connected successfully")
        except Exception as e:
            raise RuntimeError(f"❌ vLLM connection failed: {e}")
        
        print("✅ Lumber QA System initialized")
    
    def _check_vllm_server(self):
        """Check vLLM server status"""
        try:
            response = requests.get("http://localhost:8888/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def parse_json_response(self, response):
        """Parse JSON response from LLM"""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json_repair.loads(json_str)
                return parsed
            except:
                pass
        
        try:
            parsed = json_repair.loads(response)
            return parsed
        except:
            return {"answer": "UNKNOWN"}
    
    def process_single_question(self, sample, topk_values, verbose=True):
        """Process single question: encode once, retrieve once, then deterministic QA for different topk (single deterministic run)"""
        question = sample['question']
        question_id = sample['_id']
        domain = sample['sub_domain'].lower()
        choices = {
            'A': sample['choice_A'],
            'B': sample['choice_B'], 
            'C': sample['choice_C'],
            'D': sample['choice_D']
        }
        chunks = sample['chunks']
        
        if verbose:
            print(f"\n📝 Question ID: {question_id}")
            print(f"\n📝 Question: {question[:100]}...")
            print(f"🏷️ Domain: {domain}")
            print(f"📦 Available chunks: {len(chunks)}")

        if verbose:
            print(f"    🔍 Start encoding question...")
        question_embedding = self.embedding_model.encode([question], batch_size=1)

        if verbose:
            print(f"    📦 Encoding {len(chunks)} chunks...")
        start_time = time.perf_counter()
        chunk_embeddings = self.embedding_model.encode(chunks)
        encoding_time = time.perf_counter() - start_time

        total_time = float(sample.get('time', 0.0)) + float(encoding_time)
        if verbose:
            print(f"    ✅ Chunks encoding finished, total time: {total_time:.2f}s (chunking+encoding)")

        chunks_encoding_stats = {
            'question_id': question_id,
            'time': total_time
        }

        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for topk in topk_values:
            topk_indices = sorted_indices[:topk]
            relevant_chunks = [chunks[idx] for idx in topk_indices]
            similarity_scores = [float(similarities[idx]) for idx in topk_indices]

            context = "\n\n".join(relevant_chunks)

            context = self.llm.truncate_retrieved_context(context, question)
            
            prompt = f"""Based on the following document content, please answer the multiple choice question.

Document Content:
{context}

Question: {question}

Options:
A. {choices['A']}
B. {choices['B']}
C. {choices['C']}
D. {choices['D']}

Please carefully analyze the document content and select the correct answer. Respond in JSON format with the following structure:
{{
    "answer": "A/B/C/D"
}}"""

            response = self.llm.chat(prompt)
            parsed_response = self.parse_json_response(response)
            if isinstance(parsed_response, dict):
                model_answer = parsed_response.get('answer', 'UNKNOWN')
            else:
                model_answer = 'UNKNOWN'

            original_chunking_stats = {
                'chunking_time': sample.get('time', 0.0)
            }

            result = {
                'question_id': question_id,
                'question': question,
                'domain': domain,
                'topk': topk,
                'retrieved_chunks': [{'text': chunk, 'similarity': sim} for chunk, sim in zip(relevant_chunks, similarity_scores)],
                'original_chunking_stats': original_chunking_stats,
                'chunks_encoding_stats': chunks_encoding_stats,
                'raw_response': response,
                'parsed_response': parsed_response,
                'model_answer': model_answer,
                'correct_answer': sample['answer'],
                'choices': choices,
                'is_correct': model_answer.upper() == sample['answer'].upper()
            }

            results.append(result)
            if verbose:
                print(f"      TopK={topk}: {'✅ Correct' if result['is_correct'] else '❌ Incorrect'} (Ans: {model_answer} / Ref: {sample['answer']})")

        return results
    
    def evaluate_single_run(self, topk_values=[1, 3, 5], verbose=False, save_results=True):
        """Evaluate single run - Test all samples (deterministic output)"""
        print("🎯 Deterministic evaluation mode (do_sample=False, temperature=0.0)")
        print(f"🎯 TopK values: {topk_values}")

        all_results = []

        total_questions = 0
        total_time = 0
        domain_stats = {}

        for domain_name in self.datasets:
            if verbose:
                print(f"\n📋 Processing domain: {domain_name}")
            
            domain_dataset = self.datasets[domain_name]
            total_samples = len(domain_dataset)
            total_questions += total_samples
            
            print(f"    Processing {domain_name}: {total_samples} samples")

            domain_stats[domain_name] = {
                'total_samples': total_samples,
                'total_time': 0,
                'topk_stats': {topk: {'correct': 0, 'total': 0} for topk in topk_values}
            }

            for i in tqdm(range(total_samples), desc=f"Processing {domain_name}", unit="sample"):
                sample = domain_dataset[i]

                sample_results = self.process_single_question(sample, topk_values, verbose=False)

                if sample_results:
                    sample_time = sample_results[0].get('chunks_encoding_stats', {}).get('time', 0.0)
                    total_time += sample_time
                    domain_stats[domain_name]['total_time'] += sample_time

                all_results.extend(sample_results)

                for result in sample_results:
                    topk = result['topk']
                    is_correct = result['is_correct']
                    domain_stats[domain_name]['topk_stats'][topk]['total'] += 1
                    if is_correct:
                        domain_stats[domain_name]['topk_stats'][topk]['correct'] += 1
            
            print(f"  ✅ {domain_name} finished: {total_samples}/{total_samples} (100%)")

        print(f"\nTotal questions: {total_questions}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per question: {total_time/total_questions:.2f}s")

        overall_stats = {topk: {'correct': 0, 'total': 0} for topk in topk_values}
        
        for domain_name, stats in domain_stats.items():
            for topk in topk_values:
                correct = stats['topk_stats'][topk]['correct']
                total = stats['topk_stats'][topk]['total']

                overall_stats[topk]['correct'] += correct
                overall_stats[topk]['total'] += total

        print(f"\n🎯 Overall Accuracy:")
        for topk in topk_values:
            correct = overall_stats[topk]['correct']
            total = overall_stats[topk]['total']
            accuracy = correct / total if total > 0 else 0
            print(f"  TopK-{topk}: {accuracy*100:.2f}%±0.00% ({correct}/{total})")

        print(f"\n📋 Detailed statistics by domain:")
        for domain_name, stats in domain_stats.items():
            print(f"  {domain_name}:")
            print(f"    Questions: {stats['total_samples']}")
            print(f"    Total time: {stats['total_time']:.2f}s")
            print(f"    Avg time: {stats['total_time']/stats['total_samples']:.2f}s")
            for topk in topk_values:
                correct = stats['topk_stats'][topk]['correct']
                total = stats['topk_stats'][topk]['total']
                accuracy = correct / total if total > 0 else 0
                print(f"    TopK-{topk}: {accuracy*100:.2f}% ({correct}/{total})")

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"lumber_qa_results_{timestamp}.md"
            lines = []
            lines.append(f"# Lumber QA Results ({timestamp})")
            lines.append("")
            lines.append("## Overall Accuracy")
            lines.append("")
            lines.append("| TopK | Correct | Total | Accuracy |")
            lines.append("| --- | ---: | ---: | ---: |")
            for topk in topk_values:
                correct = overall_stats[topk]['correct']
                total = overall_stats[topk]['total']
                accuracy = correct / total if total > 0 else 0
                lines.append(f"| {topk} | {correct} | {total} | {accuracy*100:.2f}% |")
            lines.append("")
            lines.append("## Detailed Statistics by Domain")
            headers = ["Domain", "Questions", "Total Time(s)", "Avg Time(s)"]
            for topk in topk_values:
                headers.append(f"TopK-{topk} Correct/Total")
                headers.append(f"TopK-{topk} Accuracy")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for domain_name, stats in domain_stats.items():
                avg_time = stats['total_time'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
                row = [
                    domain_name,
                    str(stats['total_samples']),
                    f"{stats['total_time']:.2f}",
                    f"{avg_time:.2f}",
                ]
                for topk in topk_values:
                    correct = stats['topk_stats'][topk]['correct']
                    total = stats['topk_stats'][topk]['total']
                    acc = (correct / total) if total > 0 else 0
                    row.append(f"{correct}/{total}")
                    row.append(f"{acc*100:.2f}%")
                lines.append("| " + " | ".join(row) + " |")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            print(f"\n💾 Results saved as Markdown table: {results_file}")
        
        return all_results, domain_stats

def check_vllm_server():
    """Check vLLM server availability"""
    print("🔍 Checking vLLM server status...")
    try:
        response = requests.get("http://localhost:8888/health", timeout=5)
        if response.status_code == 200:
            print("✅ vLLM server is running normally")
            return True
    except:
        pass
    
    print("❌ vLLM server unavailable!")
    print("📋 Please start the server with:")
    print("   1. Run: sbatch server.sh")
    print("   2. Wait for server startup")
    print("   3. Check status: curl http://localhost:8888/health")
    raise RuntimeError("vLLM server unavailable, please start server.py first")

def main():
    print("🎯 Lumber QA System - Deterministic Evaluation")
    print("=" * 60)
    check_vllm_server()
    topk_values = [5, 10]
    embed_models = [
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v2-small-en",
        "nomic-ai/nomic-embed-text-v1.5",
    ]
    repeats = 3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"lumber_qa_summary_{timestamp}.md"
    markdown_buffer = io.StringIO()
    
    for embed_model in embed_models:
        print("\n🚀 Initializing system...")
        print(f"\n📋 Evaluation Config:")
        print(f"   TopK values: {topk_values}")
        print(f"   Chunking method: Lumber")
        print(f"   Embedding model: {embed_model}")
        domain_stats_runs = []
        for _ in range(repeats):
            lumber_dataset_path = "../Data/LongBench-v2_chunked/Lumber/qwen3-8b"
            qa_system = QASystem(
                lumber_dataset_path=lumber_dataset_path,
                embedding_model_path=embed_model
            )
            results, domain_stats = qa_system.evaluate_single_run(
                topk_values=topk_values,
                verbose=False,
                save_results=False
            )
            domain_stats_runs.append(domain_stats)
        domains = list(domain_stats_runs[0].keys())
        total_questions_runs = [sum(d.get('total_questions', d.get('total_samples', 0)) for d in r.values()) for r in domain_stats_runs]
        total_time_runs = [sum(d['total_time'] for d in r.values()) for r in domain_stats_runs]
        topk_acc_runs = {k: [] for k in topk_values}
        for r in domain_stats_runs:
            for k in topk_values:
                corr = sum(d['topk_stats'][k]['correct'] for d in r.values())
                tot = sum(d['topk_stats'][k]['total'] for d in r.values())
                acc = (corr / tot) if tot > 0 else 0.0
                topk_acc_runs[k].append(acc)
        mq = statistics.mean(total_questions_runs)
        mt = statistics.mean(total_time_runs)
        st = statistics.stdev(total_time_runs) if len(total_time_runs) > 1 else 0.0
        avg_list = [t / q if q > 0 else 0.0 for t, q in zip(total_time_runs, total_questions_runs)]
        ma = statistics.mean(avg_list)
        sa = statistics.stdev(avg_list) if len(avg_list) > 1 else 0.0

        stats_buffer = io.StringIO()
        def log_stats(msg):
            print(msg) 
            print(msg, file=stats_buffer) 
        
        log_stats(f"\n{'='*60}")
        log_stats(f"📊 Lumber Evaluation Stats ({repeats} runs mean±std)")
        log_stats(f"{'='*60}")
        log_stats(f"Embedding model: {embed_model}")
        log_stats(f"Total Questions: {int(mq)}")
        log_stats(f"Total Time: {mt:.2f}s±{st:.2f}s")
        log_stats(f"Avg Time per Question: {ma:.2f}s±{sa:.2f}s")
        log_stats(f"\n🎯 Overall Accuracy:")
        for k in topk_values:
            macc = statistics.mean(topk_acc_runs[k])
            sacc = statistics.stdev(topk_acc_runs[k]) if len(topk_acc_runs[k]) > 1 else 0.0
            log_stats(f"  TopK-{k}: {macc*100:.2f}%±{sacc*100:.2f}%")
        log_stats(f"\n📋 Detailed Stats by Domain:")
        for domain in domains:
            tts = [r[domain]['total_time'] for r in domain_stats_runs]
            tqs = [r[domain].get('total_questions', r[domain].get('total_samples', 0)) for r in domain_stats_runs]
            ats = [t / q if q > 0 else 0.0 for t, q in zip(tts, tqs)]
            mt_domain = statistics.mean(tts)
            st_domain = statistics.stdev(tts) if len(tts) > 1 else 0.0
            ma_domain = statistics.mean(ats)
            sa_domain = statistics.stdev(ats) if len(ats) > 1 else 0.0
            log_stats(f"  {domain}:")
            log_stats(f"    Questions: {int(statistics.mean(tqs))}")
            log_stats(f"    Total Time: {mt_domain:.2f}s±{st_domain:.2f}s")
            log_stats(f"    Avg Time: {ma_domain:.2f}s±{sa_domain:.2f}s")
            for k in topk_values:
                accs = []
                for r in domain_stats_runs:
                    c = r[domain]['topk_stats'][k]['correct']
                    t = r[domain]['topk_stats'][k]['total']
                    accs.append((c / t) if t > 0 else 0.0)
                macc_d = statistics.mean(accs)
                sacc_d = statistics.stdev(accs) if len(accs) > 1 else 0.0
                log_stats(f"    TopK-{k}: {macc_d*100:.2f}%±{sacc_d*100:.2f}%")
        
        markdown_buffer.write(stats_buffer.getvalue())
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(markdown_buffer.getvalue())
    print(f"\n💾 Results saved as Markdown: {summary_file}")

if __name__ == "__main__":
    main()
