
from utils.qa_pipeline import StandardQASystem, aggregate_and_print_summary
from datetime import datetime
import io

def main():
    print("🎯 Lumber QA System - Deterministic Evaluation")
    print("=" * 60)
    
    # We rely on BaseQASystem to check vLLM, but if we want to check before loop:
    # check_vllm_server() # Optional, as BaseQASystem checks it on init
    
    topk_values = [5]
    embed_models = [
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v2-small-en",
        "nomic-ai/nomic-embed-text-v1.5",
    ]
    repeats = 3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"lumber_qa_summary_{timestamp}.md"
    
    for embed_model in embed_models:
        print("\n🚀 Initializing system...")
        print(f"\n📋 Evaluation Configuration:")
        print(f"   TopK values: {topk_values}")
        print(f"   Chunking Method: Lumber")
        print(f"   Embedding Model: {embed_model}")
        domain_stats_runs = []
        for _ in range(repeats):
            lumber_dataset_path = "Qwen/Qwen3-8B"
            # Note: test_lumber.py used device_id="2"
            qa_system = StandardQASystem(
                dataset_path=lumber_dataset_path,
                embedding_model_path=embed_model,
                device_id="0",
                system_name="Lumber"
            )
            results, domain_stats = qa_system.evaluate_single_run(
                topk_values=topk_values,
                verbose=False,
                save_results=False,
                results_prefix="lumber_qa"
            )
            domain_stats_runs.append(domain_stats)
            
        aggregate_and_print_summary(domain_stats_runs, topk_values, "Lumber", repeats, embed_model, summary_file)

if __name__ == "__main__":
    main()
