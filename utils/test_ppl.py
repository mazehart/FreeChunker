
from utils.qa_pipeline import StandardQASystem, aggregate_and_print_summary
from datetime import datetime
import io

def main():
    print("Starting PPL Chunking Test...")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    topk_values = [5]
    embed_models = [
        "jinaai/jina-embeddings-v2-small-en",
        "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v1.5",
    ]
    
    print(f"Test Parameters: topk_values={topk_values} (Deterministic Run)")
    
    ppl_dataset_path = "Qwen/Qwen2.5-1.5B-Instruct"
    repeats = 3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"ppl_qa_summary_{timestamp}.md"
    
    for embed_model in embed_models:
        print("\n🚀 Initializing system...")
        print(f"\n📋 Evaluation Configuration:")
        print(f"   TopK values: {topk_values}")
        print(f"   Chunking Method: PPL")
        print(f"   Embedding Model: {embed_model}")
        domain_stats_runs = []
        for _ in range(repeats):
            qa_system = StandardQASystem(
                dataset_path=ppl_dataset_path,
                embedding_model_path=embed_model,
                device_id="4",
                system_name="PPL"
            )
            results, domain_stats = qa_system.evaluate_single_run(
                topk_values=topk_values,
                verbose=False,
                save_results=False,
                results_prefix="ppl_qa"
            )
            domain_stats_runs.append(domain_stats)
            
        aggregate_and_print_summary(domain_stats_runs, topk_values, "PPL", repeats, embed_model, summary_file)

if __name__ == "__main__":
    main()
