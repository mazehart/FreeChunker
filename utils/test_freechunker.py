
from utils.qa_pipeline import FreeChunkerQASystem, aggregate_and_print_summary
from datetime import datetime

def main():
    print("🎯 Encoder QA System - Deterministic Evaluation (vLLM API Mode)")
    print("=" * 60)
    
    topk_values = [5, 10]
    dataset_path = '/share/home/ecnuzwx/UnifiedRAG/LongBench-v2'
    scenarios = [
        {
            'name': 'jina',
            'path': '/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/jina-embeddings-v2-small-en/jina_epoch_1'
        },
        {
            'name': 'bge-m3',
            'path': '/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/bge-m3/xlmroberta_epoch_1'
        },
        {
            'name': 'nomic-embed-text-v1.5',
            'path': '/share/home/ecnuzwx/UnifiedRAG/saved_models/2-epoch/nomic-embed-text-v1.5/xlmroberta_epoch_1'
        }
    ]
    repeats = 3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"freechunker_qa_summary_{timestamp}.md"
    
    for sc in scenarios:
        print("\n🚀 Initializing system...")
        print(f"\n📋 Evaluation Configuration:")
        print(f"   Encoder: {sc['name']}")
        print(f"   TopK values: {topk_values}")
        domain_stats_runs = []
        for _ in range(repeats):
            qa_system = FreeChunkerQASystem(
                dataset_path=dataset_path,
                encoder_model_name=sc['name'],
                encoder_model_path=sc['path'],
                device_id="7",
                system_name="FreeChunker"
            )
            results, domain_stats = qa_system.evaluate_single_run(
                topk_values=topk_values,
                verbose=False,
                save_results=False,
                results_prefix=f"{sc['name']}_qa"
            )
            domain_stats_runs.append(domain_stats)
            
        aggregate_and_print_summary(domain_stats_runs, topk_values, "FreeChunker", repeats, sc['name'], summary_file)

if __name__ == "__main__":
    main()
