
from utils.qa_pipeline import StandardQASystem, aggregate_and_print_summary
from datetime import datetime
import io

def main():
    print("🎯 Traditional QA System - Deterministic Evaluation")
    print("=" * 60)
    
    topk_values = [5, 10]
    embed_models = [
        "/share/home/FreeChunker/cache/models--jinaai--jina-embeddings-v2-small-en",
        "/share/home/FreeChunker/cache/models--BAAI--bge-m3",
        "/share/home/FreeChunker/cache/models--nomic-ai--nomic-embed-text-v1.5",
    ]
    repeats = 3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"traditional_qa_summary_{timestamp}.md"
    
    s = 256
    # Global chunk_size was used for printing in original, but here we can just log it
    print(f"Chunk Size: {s}")
    path = f"/share/home/FreeChunker/LongBench-v2_chunked/Traditional/{s}"
    
    for embed_model in embed_models:
        print("\n🚀 Initializing system...")
        print(f"\n📋 Evaluation Configuration:")
        print(f"   TopK values: {topk_values}")
        print(f"   Chunking Method: Traditional")
        print(f"   Embedding Model: {embed_model}")
        domain_stats_runs = []
        for _ in range(repeats):
            qa_system = StandardQASystem(
                dataset_path=path,
                embedding_model_path=embed_model,
                device_id="6",
                system_name="Traditional"
            )
            results, domain_stats = qa_system.evaluate_single_run(
                topk_values=topk_values,
                verbose=False,
                save_results=False,
                results_prefix="traditional_qa"
            )
            domain_stats_runs.append(domain_stats)
            
        aggregate_and_print_summary(domain_stats_runs, topk_values, f"Traditional (Size {s})", repeats, embed_model, summary_file)

if __name__ == "__main__":
    main()
