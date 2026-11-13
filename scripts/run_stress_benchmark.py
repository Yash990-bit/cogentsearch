"""
Performance and Concurrency Stress Benchmark for CogentSearch Phase 1.
Executes multi-threaded ingestion across 22 mixed documents (124+ pages)
with 5 concurrent worker threads, measuring latency, memory footprint,
and idempotent repeated ingestions.
"""

import time
import json
import tracemalloc
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Ensure root package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from cogent.ingestion.pipeline import MultimodalIngestionPipeline
from cogent.schemas.document import MultimodalDocument


CORPUS_DIR = Path("data/test_corpus")
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
RESULTS_PATH = CORPUS_DIR / "benchmark_results.json"


def run_benchmark(concurrency: int = 5, repeated_runs: int = 2) -> Dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    doc_entries = manifest["documents"]
    doc_paths = [CORPUS_DIR / d["filename"] for d in doc_entries]

    logger.info(f"Starting Stress Benchmark: {len(doc_paths)} documents, {manifest['total_pages']} pages, {concurrency} workers.")

    tracemalloc.start()
    start_wall_time = time.perf_counter()

    pipeline = MultimodalIngestionPipeline(chunk_size=800, inject_visual_summaries=True)

    results: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    retry_count = 0

    # 1. Concurrent Ingestion Job
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_doc = {executor.submit(pipeline.process_file, p): p for p in doc_paths}
        for future in as_completed(future_to_doc):
            p = future_to_doc[future]
            try:
                t0 = time.perf_counter()
                doc_obj: MultimodalDocument = future.result()
                elapsed = time.perf_counter() - t0
                success_count += 1
                retries = doc_obj.raw_metadata.get("ingestion_retries", 0)
                retry_count += retries
                results.append({
                    "filename": p.name,
                    "document_id": doc_obj.document_id,
                    "pages": doc_obj.total_pages,
                    "chunks": len(doc_obj.chunks),
                    "visuals": len(doc_obj.visual_elements),
                    "time_sec": round(elapsed, 3),
                    "status": "SUCCESS",
                })
            except Exception as e:
                failure_count += 1
                logger.error(f"Concurrent ingestion failure on {p.name}: {e}")
                results.append({
                    "filename": p.name,
                    "status": "FAILED",
                    "error": str(e),
                })

    # 2. Repeated Ingestion of identical documents to verify deterministic doc_id & idempotency
    repeated_success = 0
    idempotency_mismatches = 0
    for doc_info in results:
        if doc_info.get("status") == "SUCCESS":
            p = CORPUS_DIR / doc_info["filename"]
            doc_second_run = pipeline.process_file(p)
            if doc_second_run.document_id == doc_info["document_id"]:
                repeated_success += 1
            else:
                idempotency_mismatches += 1

    total_time = time.perf_counter() - start_wall_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_pages_processed = manifest["total_pages"]
    avg_page_time = total_time / total_pages_processed if total_pages_processed > 0 else 0.0

    summary = {
        "concurrency_workers": concurrency,
        "total_documents": len(doc_paths),
        "total_pages": total_pages_processed,
        "total_processing_time_sec": round(total_time, 2),
        "avg_time_per_page_sec": round(avg_page_time, 4),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate_percent": round((success_count / len(doc_paths)) * 100, 2),
        "retry_count": retry_count,
        "repeated_idempotency_checks": repeated_success,
        "idempotency_mismatches": idempotency_mismatches,
        "per_document_metrics": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Benchmark Complete! Total Time: {summary['total_processing_time_sec']}s | "
        f"Avg/Page: {summary['avg_time_per_page_sec']}s | Peak Mem: {summary['peak_memory_mb']}MB | "
        f"Success Rate: {summary['success_rate_percent']}%"
    )
    return summary


if __name__ == "__main__":
    run_benchmark(concurrency=5)
