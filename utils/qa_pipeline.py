
import torch
import numpy as np
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import json_repair
from utils.monitor import Monitor
import requests
import time
from tqdm import tqdm
from utils.vllm_client import VLLMClient
from utils.eval_framework import truncate_chunks_by_topk, build_mcq_prompt

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
        try:
            print(f"📏 模型最大序列长度: {self.model.max_seq_length}")
            print(f"🎯 模型维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception:
            pass
    
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

class BaseQASystem:
    def __init__(self, dataset_path, device_id="0", system_name="Base"):
        print(f"🚀 正在初始化{system_name}问答系统...")
        
        # 加载数据集
        print(f"📊 正在加载数据集: {dataset_path}")
        self.datasets = load_from_disk(dataset_path)
        print(f"✅ 数据集加载完成，包含领域: {list(self.datasets.keys())}")
        
        # 初始化监控器
        self.monitor = Monitor(device_id=device_id)
        self.monitor.setup()
        
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
            
        self.system_name = system_name
        print(f"✅ {system_name}问答系统初始化完成")

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
        raise NotImplementedError("Subclasses must implement process_single_question")

    def evaluate_single_run(self, topk_values=[1, 3, 5], verbose=False, save_results=True, results_prefix="qa"):
        """单次评估运行 - 测试所有样本（确定性输出）"""
        print(f"🎯 {self.system_name} 确定性评估模式")
        print(f"🎯 TopK值列表: {topk_values}")
        
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
            total_questions += total_samples
            
            print(f"    处理{domain_name}: {total_samples}个样本")
            
            # 初始化领域统计
            domain_stats[domain_name] = {
                'total_questions': total_samples, # Unify key name
                'total_samples': total_samples,   # Keep both for compatibility if needed
                'total_time': 0.0,
                'topk_stats': {topk: {'correct': 0, 'total': 0} for topk in topk_values}
            }
            
            # 处理该领域的所有样本
            for i in tqdm(range(total_samples), desc=f"Processing {domain_name}", unit="sample"):
                sample = domain_dataset[i]
                
                # 对该样本进行处理
                sample_results = self.process_single_question(sample, topk_values, verbose=False)
                
                # 将结果直接添加到总结果中
                all_results.extend(sample_results)
                
                # 计算该样本的时间（从chunks_encoding_stats中获取）
                if sample_results:
                    # Try to get time from chunks_encoding_stats, fallback to other means if needed
                    # StandardQASystem puts 'time' in chunks_encoding_stats
                    # FreeChunker puts 'encoding_time' in chunks_encoding_stats and also has 'original_chunking_stats'
                    
                    stats = sample_results[0].get('chunks_encoding_stats', {})
                    sample_time = stats.get('time', stats.get('encoding_time', 0.0))
                    
                    # For FreeChunker, add original time if not already included
                    if 'encoding_time' in stats: 
                         original_time = float(sample.get('time', 0.0))
                         sample_time += original_time
                         
                    total_time += sample_time
                    domain_stats[domain_name]['total_time'] += sample_time
                
                # 更新正确率统计
                for result in sample_results:
                    topk = result['topk']
                    is_correct = result['is_correct']
                    domain_stats[domain_name]['topk_stats'][topk]['total'] += 1
                    if is_correct:
                        domain_stats[domain_name]['topk_stats'][topk]['correct'] += 1
            
            print(f"  ✅ {domain_name} 完成: {total_samples}/{total_samples} (100%)")
        
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
        if verbose or True: # Always print summary at end of run
            print(f"\n{'='*60}")
            print(f"📊 {self.system_name} 评估结果统计")
            print(f"{'='*60}")
            print(f"总问题数: {total_questions}")
            print(f"总耗时: {total_time:.2f}s")
            print(f"平均每题耗时: {overall_stats['avg_time_per_question']:.2f}s")
            
            print(f"\n🎯 总体准确率:")
            for topk in topk_values:
                stats = overall_stats['topk_accuracy'][topk]
                print(f"  TopK-{topk}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
            
            print(f"\n📋 各领域详细统计:")
            for domain, stats in domain_stats.items():
                print(f"  {domain}:")
                print(f"    问题数: {stats['total_questions']}")
                print(f"    总耗时: {stats['total_time']:.2f}s")
                avg_time = stats['total_time']/stats['total_questions'] if stats['total_questions'] > 0 else 0
                print(f"    平均耗时: {avg_time:.2f}s")
                for topk in topk_values:
                    topk_stat = stats['topk_stats'][topk]
                    accuracy = topk_stat['correct'] / topk_stat['total'] if topk_stat['total'] > 0 else 0
                    print(f"    TopK-{topk}: {accuracy:.2%} ({topk_stat['correct']}/{topk_stat['total']})")
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"{results_prefix}_results_{timestamp}.md"
            self._save_markdown_results(results_file, topk_values, overall_stats, domain_stats, timestamp)
            print(f"\n💾 结果已保存为Markdown表格: {results_file}")
        
        return all_results, domain_stats

    def _save_markdown_results(self, filename, topk_values, overall_stats, domain_stats, timestamp):
        lines = []
        lines.append(f"# {self.system_name} 问答结果 ({timestamp})")
        lines.append("")
        lines.append("## 总体准确率")
        lines.append("")
        lines.append("| TopK | Correct | Total | Accuracy |")
        lines.append("| --- | ---: | ---: | ---: |")
        for topk in topk_values:
            stats = overall_stats['topk_accuracy'][topk]
            lines.append(f"| {topk} | {stats['correct']} | {stats['total']} | {stats['accuracy']*100:.2f}% |")
        lines.append("")
        lines.append("## 各领域详细统计")
        headers = ["领域", "问题数", "总耗时(s)", "平均耗时(s)"]
        for topk in topk_values:
            headers.append(f"TopK-{topk} 正确/总数")
            headers.append(f"TopK-{topk} 准确率")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for domain_name, stats in domain_stats.items():
            avg_time = stats['total_time'] / stats['total_questions'] if stats['total_questions'] > 0 else 0
            row = [
                domain_name,
                str(stats['total_questions']),
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
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

def aggregate_and_print_summary(domain_stats_runs, topk_values, system_name, repeats, model_name, summary_file):
    """Aggregates results from multiple runs and prints/saves a summary."""
    import statistics
    import io
    
    domains = list(domain_stats_runs[0].keys())
    # Handle key differences: some use 'total_questions', some 'total_samples' (I unified to total_questions in BaseQASystem but let's be safe)
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
    log_stats(f"📊 {system_name}评估结果统计（{repeats}次平均±标准差）")
    log_stats(f"{'='*60}")
    log_stats(f"Embedding模型/编码器: {model_name}")
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
    
    # Append to summary file
    with open(summary_file, 'a', encoding='utf-8') as f: # Use 'a' to append
        f.write(stats_buffer.getvalue())
    print(f"\n💾 结果已追加到Markdown: {summary_file}")


class StandardQASystem(BaseQASystem):
    def __init__(self, dataset_path, embedding_model_path, device_id="0", system_name="Standard"):
        super().__init__(dataset_path, device_id, system_name)
        # 初始化embedding模型
        self.embedding_model = EmbeddingModel(embedding_model_path)

    def process_single_question(self, sample, topk_values, verbose=True):
        """处理单个问题：编码一次，检索一次，然后对不同topk进行问答"""
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
        
        # 2. 编码所有chunks
        if verbose:
            print(f"    📦 编码 {len(chunks)} 个chunks...")
        start_time = time.perf_counter()
        chunk_embeddings = self.embedding_model.encode(chunks)
        encoding_time = time.perf_counter() - start_time
        
        # 记录chunks编码统计（合并分块时间与编码时间）
        # 注意：这里假设sample中有'time'字段表示分块时间
        total_time = float(sample.get('time', 0.0)) + float(encoding_time)
        
        chunks_encoding_stats = {
            'question_id': question_id,
            'time': total_time
        }
        
        # 3. 计算一次相似度与排序
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        # 4. 对每个topk进行问答
        results = []
        for topk in topk_values:
            topk_indices = sorted_indices[:topk]
            relevant_chunks = [chunks[idx] for idx in topk_indices]
            similarity_scores = [float(similarities[idx]) for idx in topk_indices]

            context = truncate_chunks_by_topk(self.llm.tokenizer, relevant_chunks)
            prompt = build_mcq_prompt(context, question, choices)

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

from src.encoder import UnifiedEncoder

class FreeChunkerQASystem(BaseQASystem):
    def __init__(self, dataset_path, encoder_model_name, encoder_model_path, device_id="7", system_name="FreeChunker"):
        # Initialize base without loading embedding model locally in the same way
        super().__init__(dataset_path, device_id, system_name)
        
        # 初始化统一编码器
        print(f"🔧 正在初始化统一编码器: {encoder_model_name}")
        self.encoder_model_name = encoder_model_name
        self.encoder = UnifiedEncoder(
            model_name=encoder_model_name,
            local_model_path=encoder_model_path
        )
        
    def process_single_question(self, sample, topk_values, verbose=True):
        """处理单个问题：编码一次，检索一次，然后对不同topk进行确定性问答"""
        question = sample['question']
        question_id = sample['_id']
        domain = sample['sub_domain'].lower()
        choices = {
            'A': sample['choice_A'],
            'B': sample['choice_B'], 
            'C': sample['choice_C'],
            'D': sample['choice_D']
        }
        context = sample['context']
        
        if verbose:
            print(f"\n📝 问题ID: {question_id}")
            print(f"📝 问题: {question[:100]}...")
            print(f"🏷️ 领域: {domain}")
        
        start_time = time.perf_counter()
        self.encoder.build_vector_store(context, show_progress=False)
        run_time = time.perf_counter() - start_time
        
        # 记录chunks编码统计（仅时间）
        chunks_encoding_stats = {
            'question_id': question_id,
            'encoding_time': run_time
        }
        
        # 获取原始分块时间
        original_chunking_stats = {
            'chunking_time': sample.get('time', 0.0)
        }
        
        # 2. 对每个topk值分别进行检索和LLM问答
        results = []
        for topk in topk_values:
            # 对当前topk值进行检索
            retrieved_context = self.encoder.query(
                query=question,
                top_k=topk,
                aggregation_mode='post',
                tokenizer=self.llm.tokenizer
            )
            
            prompt = build_mcq_prompt(retrieved_context, question, choices)

            # 调用LLM生成答案（确定性运行）
            response = self.llm.chat(prompt)
            
            # 解析JSON响应
            parsed_response = self.parse_json_response(response)
            
            if isinstance(parsed_response, dict):
                model_answer = parsed_response.get('answer', 'UNKNOWN')
            else:
                model_answer = 'UNKNOWN'
            
            # 保存结果
            result = {
                'question_id': question_id,
                'question': question,
                'domain': domain,
                'topk': topk,
                'retrieved_context': retrieved_context,
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
            
            # 显示每个回答的对错
            if verbose:
                print(f"      TopK={topk}: {'✅ 正确' if result['is_correct'] else '❌ 错误'} (答案: {model_answer} / 正确: {sample['answer']})")
        
        return results

