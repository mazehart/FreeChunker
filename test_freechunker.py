from datasets import load_from_disk
import re
from datetime import datetime
import json_repair
from utils.monitor import Monitor
import time
from src.encoder import UnifiedEncoder
from openai import OpenAI
import requests
from transformers import AutoTokenizer
from tqdm import tqdm
import io
import statistics

monitor = Monitor(device_id="5")
monitor.setup()

class VLLMClient:
    def __init__(self, system_prompt="You are an excellent reading comprehension assistant. Please provide answers in JSON format.", do_sample=False, temperature=0.7):
        """直接使用 OpenAI 客户端连接 vLLM"""
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8888/v1"
        )
        self.system_prompt = system_prompt
        self.do_sample = do_sample
        self.temperature = temperature
        self.device = "vllm-server"
        
        # 加载 Qwen tokenizer 用于精确计算 token 数量
        print("🔧 正在加载 Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B")
        self.max_context_length = 40000  # Qwen3-8B 的最大上下文长度
        self.reserved_tokens = 1000  # 为系统提示、问题和回答预留的 token 数量
        
        print(f"✅ vLLM 客户端初始化完成")
        print(f"🎲 生成参数: do_sample={do_sample}, temperature={temperature}")
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
            "temperature": 0.0,
            "extra_body": {
                "do_sample": False,
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

class EncoderQASystem:
    def __init__(self, 
                 dataset_path='/share/home/ecnuzwx/UnifiedRAG/LongBench-v2',
                 encoder_model_name='jina',
                 encoder_model_path='/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/jina-embeddings-v2-small-en/jina_epoch_1',
                 do_sample=False,
                 temperature=0.0):
        
        print("🚀 正在初始化Encoder问答系统...")
        
        # 加载数据集
        print(f"📊 正在加载数据集: {dataset_path}")
        self.datasets = load_from_disk(dataset_path)
        print(f"✅ 数据集加载完成，包含领域: {list(self.datasets.keys())}")
        
        # 初始化统一编码器
        print(f"🔧 正在初始化统一编码器: {encoder_model_name}")
        self.encoder_model_name = encoder_model_name
        self.encoder = UnifiedEncoder(
            model_name=encoder_model_name,
            local_model_path=encoder_model_path
        )
        
        # 初始化 vLLM 客户端
        print(f"🔧 正在初始化LLM (vLLM API 模式)...")
        
        # 检查服务器状态
        if not self._check_vllm_server():
            raise RuntimeError("❌ vLLM 服务器不可用！请先启动 server.py 提供的服务")
        
        try:
            self.llm = VLLMClient(
                do_sample=do_sample, 
                temperature=temperature
            )
            print("✅ vLLM API 客户端连接成功")
        except Exception as e:
            raise RuntimeError(f"❌ vLLM 连接失败: {e}")
        
        print("✅ Encoder问答系统初始化完成")
        
    
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
        """处理单个问题：编码一次，检索一次，然后对不同topk进行确定性问答
        
        Args:
            sample: 问题样本
            topk_values: topk值列表，例如 [1, 3, 5]
            verbose: 是否显示详细信息
        """
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
            retrieved_context = self.encoder.query(question, topk)
            
            # 截断检索内容以确保不超过最大上下文长度
            retrieved_context = self.llm.truncate_retrieved_context(retrieved_context, question)
            
            # 构建prompt
            prompt = f"""Based on the following document content, please answer the multiple choice question.

Document Content:
{retrieved_context}

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
    
    def evaluate_single_run(self, topk_values=[1, 3, 5], verbose=False, save_results=True):
        """评估所有样本 - 确定性单次运行"""
        
        print("🎯 运行确定性评估模式")
        print(f"🎯 TopK值列表: {topk_values}")
        
        # 计算总样本数
        total_samples_all = sum(len(self.datasets[domain]) for domain in self.datasets)
        print(f"📊 总样本数: {total_samples_all}")
        print(f"📊 每个样本将进行: {len(topk_values)} 个topk 次问答")
        
        # 存储所有结果
        all_results = []
        processed_count = 0
        
        # 统计信息
        total_questions = 0
        total_time = 0.0
        domain_stats = {}
        
        # 按领域处理
        for domain_idx, domain_name in enumerate(self.datasets, 1):
            domain_dataset = self.datasets[domain_name]
            total_samples = len(domain_dataset)
            
            print(f"\n{'='*80}")
            print(f"📋 领域 [{domain_idx}/{len(self.datasets)}]: {domain_name} ({total_samples}个样本)")
            print('='*80)
            
            # 初始化领域统计
            domain_stats[domain_name] = {
                'total_questions': total_samples,
                'total_time': 0.0,
                'topk_stats': {topk: {'correct': 0, 'total': 0} for topk in topk_values}
            }
            
            # 处理该领域的所有样本
            for i in tqdm(range(total_samples), desc=f"Processing {domain_name}", unit="sample"):
                sample = domain_dataset[i]
                processed_count += 1
                
                # 对该样本进行一次编码、一次检索，然后确定性问答
                sample_results = self.process_single_question(sample, topk_values, verbose=False)
                
                # 将结果直接添加到总结果中
                all_results.extend(sample_results)
                
                # 更新统计信息
                total_questions += 1
                
                # 计算该样本的时间（从chunks_encoding_stats中获取）
                if sample_results:
                    sample_time = sample_results[0].get('chunks_encoding_stats', {}).get('encoding_time', 0.0)
                    # 加上原始分块时间（如果有）
                    original_time = float(sample.get('time', 0.0))
                    total_sample_time = sample_time + original_time
                    
                    total_time += total_sample_time
                    domain_stats[domain_name]['total_time'] += total_sample_time
                    
                    # 将总时间注入结果
                    for res in sample_results:
                        res['total_sample_time'] = total_sample_time
                
                # 更新正确率统计
                for result in sample_results:
                    topk = result['topk']
                    is_correct = result['is_correct']
                    domain_stats[domain_name]['topk_stats'][topk]['total'] += 1
                    if is_correct:
                        domain_stats[domain_name]['topk_stats'][topk]['correct'] += 1
            
            # 打印当前领域的统计结果
            if verbose or True:  # 强制打印
                print(f"\n📊 领域 {domain_name} 评估结果:")
                stats = domain_stats[domain_name]
                print(f"    问题数: {stats['total_questions']}")
                print(f"    总耗时: {stats['total_time']:.2f}s")
                avg_time = stats['total_time']/stats['total_questions'] if stats['total_questions'] > 0 else 0
                print(f"    平均耗时: {avg_time:.2f}s")
                for topk in topk_values:
                    topk_stat = stats['topk_stats'][topk]
                    accuracy = topk_stat['correct'] / topk_stat['total'] if topk_stat['total'] > 0 else 0
                    print(f"    TopK-{topk}: {accuracy:.2%} ({topk_stat['correct']}/{topk_stat['total']})")
                print("-" * 60)
        
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
            print(f"📊 Qwen3编码器评估结果统计")
            print(f"{'='*60}")
            print(f"总问题数: {total_questions}")
            print(f"总耗时: {total_time:.2f}s")
            print(f"平均每题耗时: {overall_stats['avg_time_per_question']:.2f}s")
            
            print(f"\n🎯 总体准确率:")
            for topk in topk_values:
                stats = overall_stats['topk_accuracy'][topk]
                print(f"  TopK-{topk}: {stats['accuracy']:.2%}±{0:.2%} ({stats['correct']}/{stats['total']})")
            
            print(f"\n📋 各领域详细统计:")
            for domain, stats in domain_stats.items():
                print(f"  {domain}:")
                print(f"    问题数: {stats['total_questions']}")
                print(f"    总耗时: {stats['total_time']:.2f}s")
                print(f"    平均耗时: {stats['total_time']/stats['total_questions']:.2f}s")
                for topk in topk_values:
                    topk_stat = stats['topk_stats'][topk]
                    accuracy = topk_stat['correct'] / topk_stat['total'] if topk_stat['total'] > 0 else 0
                    print(f"    TopK-{topk}: {accuracy:.2%} ({topk_stat['correct']}/{topk_stat['total']})")
        
        print(f"\n{'='*80}")
        print(f"✅ 所有样本处理完成: {processed_count}/{total_samples_all}")
        print(f"✅ 总问答次数: {len(all_results)}")
        print('='*80)
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"{self.encoder_model_name}_qa_results_{timestamp}.md"
            lines = []
            lines.append(f"# {self.encoder_model_name} 编码器问答结果 ({timestamp})")
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
    print("🎯 Encoder问答系统 - 确定性评估 (vLLM API 模式)")
    print("=" * 60)
    check_vllm_server()
    topk_values = [5, 10]
    dataset_path = '/share/home/ecnuzwx/UnifiedRAG/LongBench-v2'
    scenarios = [
        {
            'name': 'jina',
            'path': '/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/jina-embeddings-v2-small-en/jina_epoch_1'
        },
        {
            'name': 'nomic-embed-text-v1.5',
            'path': '/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/nomic-embed-text-v1.5/xlmroberta_epoch_1'
        },
        {
            'name': 'bge-m3',
            'path': '/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/bge-m3/xlmroberta_epoch_1'
        },
    ]
    repeats = 1
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"freechunker_qa_summary_{timestamp}.md"
    markdown_buffer = io.StringIO()
    
    for sc in scenarios:
        print("\n🚀 初始化系统...")
        print(f"\n📋 评估配置:")
        print(f"   编码器: {sc['name']}")
        print(f"   TopK值: {topk_values}")
        domain_stats_runs = []
        for _ in range(repeats):
            qa_system = EncoderQASystem(
                dataset_path=dataset_path,
                encoder_model_name=sc['name'],
                encoder_model_path=sc['path'],
                do_sample=False,
                temperature=0.0
            )
            results, domain_stats = qa_system.evaluate_single_run(topk_values=topk_values, verbose=False, save_results=False)
            domain_stats_runs.append(domain_stats)
        domains = list(domain_stats_runs[0].keys())
        total_questions_runs = [sum(d['total_questions'] for d in r.values()) for r in domain_stats_runs]
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
        log_stats(f"📊 FreeChunker评估结果统计（{repeats}次平均±标准差）")
        log_stats(f"{'='*60}")
        log_stats(f"编码器: {sc['name']}")
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
            tqs = [r[domain]['total_questions'] for r in domain_stats_runs]
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
