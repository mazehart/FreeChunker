
import torch
import numpy as np
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from datetime import datetime
import json_repair
from utils.monitor import Monitor
from openai import OpenAI
import requests
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import io
import sys
import statistics

chunk_size = 512

class EmbeddingModel:
    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer
        print(f"🔧 正在加载embedding模型: {model_path}")
        
        # 使用sentence-transformers，自动处理批次和显存管理
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式，禁用dropout等训练相关层
        
        print(f"✅ Embedding模型加载完成，设备: {self.device}")
        print(f"📏 模型最大序列长度: {self.model.max_seq_length}")
        print(f"🎯 模型维度: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts, batch_size=6):
        """编码文本"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化embedding
        )
        
        return embeddings

class VLLMClient:
    def __init__(self, system_prompt="You are an excellent reading comprehension assistant. Please provide answers in JSON format."):
        """直接使用 OpenAI 客户端连接 vLLM"""
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8888/v1"
        )
        self.system_prompt = system_prompt
        self.device = "vllm-server"
        
        # 加载 Qwen tokenizer 用于精确计算 token 数量
        print("🔧 正在加载 Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B")
        self.max_context_length = 40000  # Qwen3-8B 的最大上下文长度
        self.reserved_tokens = 1000  # 为系统提示、问题和回答预留的 token 数量
        
        print(f"✅ vLLM 客户端初始化完成")
        print(f"🎲 生成参数: 确定性输出模式")
        print(f"📏 最大上下文长度: {self.max_context_length}, 预留 token: {self.reserved_tokens}")
    
    def chat(self, text, **kwargs):
        """生成回答"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        
        request_params = {
            "model": "/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0,  # 确定性输出
            "extra_body": {
                "do_sample": False,  # 确定性输出
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
        
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content.strip()
    
    def truncate_retrieved_context(self, retrieved_context, question):
        """截断检索内容以确保不超过最大上下文长度"""
        # 计算系统提示和问题的 token 数量
        system_tokens = len(self.tokenizer.encode(self.system_prompt))
        question_tokens = len(self.tokenizer.encode(question))
        
        # 计算可用于检索内容的最大 token 数量
        max_context_tokens = self.max_context_length - system_tokens - question_tokens - self.reserved_tokens
        
        # 如果检索内容的 token 数量超过限制，则进行截断
        context_tokens = self.tokenizer.encode(retrieved_context)
        if len(context_tokens) > max_context_tokens:
            print(f"⚠️  检索内容过长 ({len(context_tokens)} tokens)，截断至 {max_context_tokens} tokens")
            truncated_tokens = context_tokens[:max_context_tokens]
            retrieved_context = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return retrieved_context

class QASystem:
    def __init__(self, 
                 semantic_dataset_path='./datasets_chunked/Semantic',
                 embedding_model_path='/share/home/ecnuzwx/UnifiedRAG/cache/models--jinaai--jina-embeddings-v2-small-en'):
        
        print("🚀 正在初始化Semantic问答系统...")
        
        # 加载数据集
        print("📊 正在加载Semantic数据集...")
        self.datasets = load_from_disk(semantic_dataset_path)
        print(f"✅ 数据集加载完成，包含领域: {list(self.datasets.keys())}")
        
        # 初始化监控器
        self.monitor = Monitor(device_id="3")
        self.monitor.setup()
        
        # 初始化embedding模型
        self.embedding_model = EmbeddingModel(embedding_model_path)
        
        # 初始化 vLLM 客户端
        print(f"🔧 正在初始化LLM (vLLM API 模式)...")
        
        # 检查服务器状态
        if not self._check_vllm_server():
            raise RuntimeError("❌ vLLM 服务器不可用！请先启动 server.py 提供的服务")
        
        try:
            self.llm = VLLMClient()
            print("✅ vLLM API 客户端连接成功")
        except Exception as e:
            raise RuntimeError(f"❌ vLLM 连接失败: {e}")
        
        print("✅ Semantic问答系统初始化完成")
    
    def _check_vllm_server(self):
        """检查 vLLM 服务器状态"""
        try:
            response = requests.get("http://localhost:8888/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def parse_json_response(self, response):
        """解析LLM返回的JSON响应"""
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
        """处理单个问题：编码一次，检索一次，然后对不同topk进行问答（单次确定性运行）"""
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
            print(f"\n📝 问题ID: {question_id}")
            print(f"📝 问题: {question[:100]}...")
            print(f"🏷️ 领域: {domain}")
            print(f"📦 可用chunks: {len(chunks)}个")
        
        # 1. 问题编码（只编码一次）
        if verbose:
            print(f"    🔍 开始编码问题...")
        question_embedding = self.embedding_model.encode([question], batch_size=1)
        
        # 2. 编码所有chunks（记录资源使用）
        if verbose:
            print(f"    📦 编码 {len(chunks)} 个chunks...")
        start_time = time.perf_counter()
        chunk_embeddings = self.embedding_model.encode(chunks)
        encoding_time = time.perf_counter() - start_time
        
        # 记录chunks编码统计（合并分块时间与编码时间）
        total_time = float(sample.get('time', 0.0)) + float(encoding_time)
        if verbose:
            print(f"    ✅ Chunks编码完成，总耗时: {total_time:.2f}s (分块+编码)")
        
        # 记录chunks编码统计
        chunks_encoding_stats = {
            'question_id': question_id,
            'time': total_time
        }
        
        # 3. 计算一次相似度与排序
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        # 4. 对每个topk进行问答（单次确定性运行）
        results = []
        for topk in topk_values:
            topk_indices = sorted_indices[:topk]
            relevant_chunks = [chunks[idx] for idx in topk_indices]
            similarity_scores = [float(similarities[idx]) for idx in topk_indices]

            context = "\n\n".join(relevant_chunks)
            
            # 截断检索内容以确保不超过最大上下文长度
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
                print(f"      TopK={topk}: {'✅ 正确' if result['is_correct'] else '❌ 错误'} (答案: {model_answer} / 正确: {sample['answer']})")

        return results
    
    def evaluate_single_run(self, topk_values=[1, 3, 5], verbose=False, save_results=True):
        """单次评估运行 - 测试所有样本（确定性输出）"""
        print("🎯 确定性评估模式 (do_sample=False, temperature=0.0)")
        print(f"🎯 TopK值列表: {topk_values}")
        
        # 存储所有结果
        all_results = []
        
        # 统计信息
        total_questions = 0
        total_time = 0
        domain_stats = {}
        
        # 按领域处理
        for domain_name in self.datasets:
            if verbose:
                print(f"\n📋 处理领域: {domain_name}")
            
            domain_dataset = self.datasets[domain_name]
            total_samples = len(domain_dataset)
            total_questions += total_samples
            
            print(f"    处理{domain_name}: {total_samples}个样本")
            
            # 初始化领域统计
            domain_stats[domain_name] = {
                'total_samples': total_samples,
                'total_time': 0,
                'topk_stats': {topk: {'correct': 0, 'total': 0} for topk in topk_values}
            }
            
            # 处理该领域的所有样本
            for i in tqdm(range(total_samples), desc=f"Processing {domain_name}", unit="sample"):
                sample = domain_dataset[i]
                
                # 对该样本进行一次编码，多次检索
                sample_results = self.process_single_question(sample, topk_values, verbose=False)
                
                # 计算该样本的时间（从chunks_encoding_stats中获取）
                if sample_results:
                    sample_time = sample_results[0].get('chunks_encoding_stats', {}).get('time', 0.0)
                    total_time += sample_time
                    domain_stats[domain_name]['total_time'] += sample_time
                
                # 将结果直接添加到总结果中
                all_results.extend(sample_results)
                
                # 更新统计信息
                for result in sample_results:
                    topk = result['topk']
                    is_correct = result['is_correct']
                    domain_stats[domain_name]['topk_stats'][topk]['total'] += 1
                    if is_correct:
                        domain_stats[domain_name]['topk_stats'][topk]['correct'] += 1
            
            print(f"  ✅ {domain_name} 完成: {total_samples}/{total_samples} (100%)")
        
        # 计算总体统计
        print(f"\n总问题数: {total_questions}")
        print(f"总耗时: {total_time:.2f}s")
        print(f"平均每题耗时: {total_time/total_questions:.2f}s")
        
        # 计算总体准确率
        overall_stats = {topk: {'correct': 0, 'total': 0} for topk in topk_values}
        
        for domain_name, stats in domain_stats.items():
            for topk in topk_values:
                correct = stats['topk_stats'][topk]['correct']
                total = stats['topk_stats'][topk]['total']
                
                # 累计到总体统计
                overall_stats[topk]['correct'] += correct
                overall_stats[topk]['total'] += total
        
        # 显示总体准确率
        print(f"\n🎯 总体准确率:")
        for topk in topk_values:
            correct = overall_stats[topk]['correct']
            total = overall_stats[topk]['total']
            accuracy = correct / total if total > 0 else 0
            print(f"  TopK-{topk}: {accuracy*100:.2f}%±0.00% ({correct}/{total})")
        
        # 显示各领域详细统计
        print(f"\n📋 各领域详细统计:")
        for domain_name, stats in domain_stats.items():
            print(f"  {domain_name}:")
            print(f"    问题数: {stats['total_samples']}")
            print(f"    总耗时: {stats['total_time']:.2f}s")
            print(f"    平均耗时: {stats['total_time']/stats['total_samples']:.2f}s")
            for topk in topk_values:
                correct = stats['topk_stats'][topk]['correct']
                total = stats['topk_stats'][topk]['total']
                accuracy = correct / total if total > 0 else 0
                print(f"    TopK-{topk}: {accuracy*100:.2f}% ({correct}/{total})")
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"semantic_qa_results_{timestamp}.md"
            lines = []
            lines.append(f"# Semantic问答结果 ({timestamp})")
            lines.append("")
            lines.append("## 总体准确率")
            lines.append("")
            lines.append("| TopK | Correct | Total | Accuracy |")
            lines.append("| --- | ---: | ---: | ---: |")
            for topk in topk_values:
                correct = overall_stats[topk]['correct']
                total = overall_stats[topk]['total']
                accuracy = correct / total if total > 0 else 0
                lines.append(f"| {topk} | {correct} | {total} | {accuracy*100:.2f}% |")
            lines.append("")
            lines.append("## 各领域详细统计")
            headers = ["领域", "问题数", "总耗时(s)", "平均耗时(s)"]
            for topk in topk_values:
                headers.append(f"TopK-{topk} 正确/总数")
                headers.append(f"TopK-{topk} 准确率")
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
            print(f"\n💾 结果已保存为Markdown表格: {results_file}")
        
        return all_results, domain_stats



def check_vllm_server():
    """检查 vLLM 服务器是否可用"""
    print("🔍 检查 vLLM 服务器状态...")
    try:
        response = requests.get("http://localhost:8888/health", timeout=5)
        if response.status_code == 200:
            print("✅ vLLM 服务器运行正常")
            return True
    except:
        pass
    
    print("❌ vLLM 服务器不可用！")
    print("📋 请按以下步骤启动服务器：")
    print("   1. 运行: sbatch server.sh")
    print("   2. 等待服务器启动完成")
    print("   3. 检查服务器状态: curl http://localhost:8888/health")
    raise RuntimeError("vLLM 服务器不可用，请先启动 server.py")

def main():
    print("🎯 Semantic问答系统 - 确定性评估")
    print("=" * 60)
    check_vllm_server()
    topk_values = [5, 10]
    embed_models = [
        "/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3",
        "/share/home/ecnuzwx/UnifiedRAG/cache/models--jinaai--jina-embeddings-v2-small-en",
        "/share/home/ecnuzwx/UnifiedRAG/cache/models--nomic-ai--nomic-embed-text-v1.5",
    ]
    repeats = 3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"semantic_qa_summary_{timestamp}.md"
    markdown_buffer = io.StringIO()
    
    for embed_model in embed_models:
        print("\n🚀 初始化系统...")
        print(f"\n📋 评估配置:")
        print(f"   TopK值: {topk_values}")
        print(f"   分块方法: Semantic")
        print(f"   Embedding模型: {embed_model}")
        domain_stats_runs = []
        for _ in range(repeats):
            semantic_dataset_path = "/share/home/ecnuzwx/UnifiedRAG/LongBench-v2_chunked/Semantic/models--BAAI--bge-m3"
            qa_system = QASystem(
                semantic_dataset_path=semantic_dataset_path,
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
        
        # Capture stats to string for markdown
        stats_buffer = io.StringIO()
        def log_stats(msg):
            print(msg) # To console
            print(msg, file=stats_buffer) # To buffer
        
        log_stats(f"\n{'='*60}")
        log_stats(f"📊 Semantic评估结果统计（{repeats}次平均±标准差）")
        log_stats(f"{'='*60}")
        log_stats(f"Embedding模型: {embed_model}")
        log_stats(f"总问题数: {int(mq)}")
        log_stats(f"总耗时: {mt:.2f}s±{st:.2f}s")
        log_stats(f"平均每题耗时: {ma:.2f}s±{sa:.2f}s")
        log_stats(f"\n🎯 总体准确率:")
        for k in topk_values:
            macc = statistics.mean(topk_acc_runs[k])
            sacc = statistics.stdev(topk_acc_runs[k]) if len(topk_acc_runs[k]) > 1 else 0.0
            log_stats(f"  TopK-{k}: {macc*100:.2f}%±{sacc*100:.2f}%")
        log_stats(f"\n📋 各领域详细统计:")
        for domain in domains:
            tts = [r[domain]['total_time'] for r in domain_stats_runs]
            tqs = [r[domain].get('total_questions', r[domain].get('total_samples', 0)) for r in domain_stats_runs]
            ats = [t / q if q > 0 else 0.0 for t, q in zip(tts, tqs)]
            mt_domain = statistics.mean(tts)
            st_domain = statistics.stdev(tts) if len(tts) > 1 else 0.0
            ma_domain = statistics.mean(ats)
            sa_domain = statistics.stdev(ats) if len(ats) > 1 else 0.0
            log_stats(f"  {domain}:")
            log_stats(f"    问题数: {int(statistics.mean(tqs))}")
            log_stats(f"    总耗时: {mt_domain:.2f}s±{st_domain:.2f}s")
            log_stats(f"    平均耗时: {ma_domain:.2f}s±{sa_domain:.2f}s")
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
    print(f"\n💾 结果已保存为Markdown: {summary_file}")

if __name__ == "__main__":
    main()
