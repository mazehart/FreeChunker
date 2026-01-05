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
import io
import sys
import statistics

class EmbeddingModel:
    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer
        print(f"🔧 正在加载embedding模型: {model_path}")
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式，禁用dropout等训练相关层
        print(f"✅ Embedding模型加载完成，设备: {self.device}")
        try:
            # 仅用于信息展示，不做长度截断
            print(f"📏 模型最大序列长度: {self.model.max_seq_length}")
            print(f"🎯 模型维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception:
            pass
    
    def encode(self, texts, batch_size=4):
        
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
        """直接使用 OpenAI 客户端连接 vLLM"""
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8888/v1"
        )
        self.system_prompt = system_prompt
        self.device = "vllm-server"
        
        print(f"✅ vLLM 客户端初始化完成")
        print(f"🎲 生成参数: 确定性输出模式")
        print("🔧 正在加载 Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B")
        self.max_context_length = 40000
        self.reserved_tokens = 1000
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
            "max_tokens": 512,
            "temperature": 0.0,
            "extra_body": {
                "do_sample": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
        
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content.strip()
    
    def truncate_retrieved_context(self, retrieved_context, question):
        system_tokens = len(self.tokenizer.encode(self.system_prompt))
        question_tokens = len(self.tokenizer.encode(question))
        max_context_tokens = self.max_context_length - system_tokens - question_tokens - self.reserved_tokens
        context_tokens = self.tokenizer.encode(retrieved_context)
        if len(context_tokens) > max_context_tokens:
            truncated_tokens = context_tokens[:max_context_tokens]
            retrieved_context = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return retrieved_context

class QASystem: 
    def __init__(self, 
                 ppl_dataset_path='./datasets_chunked/PPL',
                 embedding_model_path='/share/home/ecnuzwx/UnifiedRAG/cache/models--jinaai--jina-embeddings-v2-small-en'):
        #/share/home/ecnuzwx/UnifiedRAG/cache/models--nomic-ai--nomic-embed-text-v1.5
        #/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3
        
        print("🚀 正在初始化PPL问答系统...")
        
        # 加载数据集
        print("📊 正在加载PPL数据集...")
        self.datasets = load_from_disk(ppl_dataset_path)
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
        
        print("✅ PPL问答系统初始化完成")
    
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
        
        # 2. 编码所有chunks（记录总耗时 = 分块时间 + 编码时间）
        if verbose:
            print(f"    📦 编码 {len(chunks)} 个chunks...")
        start_time = time.perf_counter()
        chunk_embeddings = self.embedding_model.encode(chunks)
        encoding_time = time.perf_counter() - start_time
        
        # 记录chunks编码统计（合并分块时间与编码时间）
        total_time = float(sample.get('time', 0.0)) + float(encoding_time)
        chunks_encoding_stats = {
            'question_id': question_id,
            'time': total_time
        }
        
        if verbose:
            print(f"    ✅ Chunks编码完成，总耗时: {total_time:.3f}s (分块+编码)")
        
        # 3. 计算一次相似度与排序
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        # 4. 对每个topk和seed进行问答
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
                print(f"      TopK={topk}: {'✅ 正确' if result['is_correct'] else '❌ 错误'} (答案: {model_answer} / 正确: {sample['answer']})")

        return results
    
    def evaluate_single_run(self, topk_values=[1, 3, 5], verbose=True, save_results=False):
        """单次评估运行 - 测试所有样本"""
        print(f"🎲 生成参数: 确定性输出模式")
        
        # 存储所有结果
        all_results = []
        
        # 统计信息
        total_questions = 0
        total_time = 0.0
        domain_stats = {}
        
        # 按领域处理
        for domain_name in self.datasets:
            if verbose:
                print(f"\n📋 处理领域: {domain_name}")
            
            domain_dataset = self.datasets[domain_name]
            total_samples = len(domain_dataset)
            
            print(f"    处理{domain_name}: {total_samples}个样本")
            
            # 初始化领域统计
            domain_stats[domain_name] = {
                'total_questions': total_samples,
                'total_time': 0.0,
                'topk_stats': {topk: {'correct': 0, 'total': 0} for topk in topk_values}
            }
            
            # 处理该领域的所有样本
            for i in tqdm(range(total_samples), desc=f"Processing {domain_name}", unit="sample"):
                sample = domain_dataset[i]
                
                # 对该样本进行一次编码，多次检索
                sample_results = self.process_single_question(sample, topk_values, verbose=False)
                
                # 将结果直接添加到总结果中
                all_results.extend(sample_results)
                
                # 更新统计信息
                total_questions += 1
                
                # 计算该样本的时间（从chunks_encoding_stats中获取）
                if sample_results:
                    sample_time = sample_results[0].get('chunks_encoding_stats', {}).get('time', 0.0)
                    total_time += sample_time
                    domain_stats[domain_name]['total_time'] += sample_time
                
                # 更新正确率统计
                for result in sample_results:
                    topk = result['topk']
                    is_correct = result['is_correct']
                    domain_stats[domain_name]['topk_stats'][topk]['total'] += 1
                    if is_correct:
                        domain_stats[domain_name]['topk_stats'][topk]['correct'] += 1
        
        # 计算总体统计
        overall_stats = {
            'total_questions': total_questions,
            'total_time': total_time,
            'avg_time_per_question': total_time / total_questions if total_questions > 0 else 0,
            'topk_accuracy': {}
        }
        
        # 计算每个TopK的总体准确率
        for topk in topk_values:
            total_correct = sum(domain_stats[domain]['topk_stats'][topk]['correct'] for domain in domain_stats)
            total_count = sum(domain_stats[domain]['topk_stats'][topk]['total'] for domain in domain_stats)
            accuracy = total_correct / total_count if total_count > 0 else 0
            overall_stats['topk_accuracy'][topk] = {
                'correct': total_correct,
                'total': total_count,
                'accuracy': accuracy
            }
        
        # 打印统计结果
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 PPL分块评估结果统计")
            print(f"{'='*60}")
            print(f"总问题数: {total_questions}")
            print(f"总耗时: {total_time:.2f}s")
            print(f"平均每题耗时: {overall_stats['avg_time_per_question']:.2f}s")
            
            print(f"\n🎯 总体准确率:")
            for topk in topk_values:
                stats = overall_stats['topk_accuracy'][topk]
                print(f"  TopK-{topk}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
            
            print(f"\n📋 各领域详细统计:")
            for domain, stats in domain_stats.items():
                print(f"  {domain}:")
                print(f"    问题数: {stats['total_questions']}")
                print(f"    总耗时: {stats['total_time']:.2f}s")
                print(f"    平均耗时: {stats['total_time']/stats['total_questions']:.2f}s")
                for topk in topk_values:
                    topk_stat = stats['topk_stats'][topk]
                    accuracy = topk_stat['correct'] / topk_stat['total'] if topk_stat['total'] > 0 else 0
                    print(f"    TopK-{topk}: {accuracy:.4f} ({topk_stat['correct']}/{topk_stat['total']})")
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"ppl_qa_results_{timestamp}.json"
            
            # 构建完整的结果数据
            results_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'method': 'PPL_chunking',
                    'topk_values': topk_values,
                    'total_questions': total_questions,
                    'generation_mode': 'deterministic'
                },
                'overall_stats': overall_stats,
                'domain_stats': domain_stats,
                'detailed_results': all_results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 结果已保存到: {results_file}")
        
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
    return False

def main():
    print("开始PPL分块测试...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查VLLM服务器
    if not check_vllm_server():
        print("VLLM服务器未启动，请先启动服务器")
        return
    
    # 测试参数
    topk_values = [5, 10]
    embed_models = [
        "/share/home/ecnuzwx/UnifiedRAG/cache/models--jinaai--jina-embeddings-v2-small-en",
        "/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3",
        "/share/home/ecnuzwx/UnifiedRAG/cache/models--nomic-ai--nomic-embed-text-v1.5",
    ]
    
    print(f"测试参数: topk_values={topk_values} (确定性运行)")
    
    ppl_dataset_path = "/share/home/ecnuzwx/UnifiedRAG/LongBench-v2_chunked/PPL/Qwen2.5-1.5B-Instruct"
    repeats = 3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"ppl_qa_summary_{timestamp}.md"
    markdown_buffer = io.StringIO()
    
    for embed_model in embed_models:
        print("\n🚀 初始化系统...")
        print(f"\n📋 评估配置:")
        print(f"   TopK值: {topk_values}")
        print(f"   分块方法: PPL")
        print(f"   Embedding模型: {embed_model}")
        domain_stats_runs = []
        for _ in range(repeats):
            qa_system = QASystem(ppl_dataset_path=ppl_dataset_path, embedding_model_path=embed_model)
            results, domain_stats = qa_system.evaluate_single_run(topk_values, verbose=False, save_results=False)
            domain_stats_runs.append(domain_stats)
        domains = list(domain_stats_runs[0].keys())
        total_questions_runs = []
        total_time_runs = []
        topk_acc_runs = {k: [] for k in topk_values}
        for stats_run in domain_stats_runs:
            tq = sum(d.get('total_questions', d.get('total_samples', 0)) for d in stats_run.values())
            tt = sum(d['total_time'] for d in stats_run.values())
            total_questions_runs.append(tq)
            total_time_runs.append(tt)
            for k in topk_values:
                corr = sum(d['topk_stats'][k]['correct'] for d in stats_run.values())
                tot = sum(d['topk_stats'][k]['total'] for d in stats_run.values())
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
        log_stats(f"📊 PPL分块评估结果统计（{repeats}次平均±标准差）")
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
