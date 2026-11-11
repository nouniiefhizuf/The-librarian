import json
import math
import time
from pathlib import Path
from src.retrieval import (get_bm25, get_collection, get_embedding_model, get_reranker, load_keyword_corpus, tokenize, KEYWORD_BOOST, STOP_WORDS,)

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_PATH = ROOT / "eval" / "retrieval_results.jsonl"
SUMMARY_PATH = ROOT / "eval" / "retrieval_summary.json"
TOP_K_VALUES = (5, 10)
RRF_K = 60

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def retrieve_candidates(question, mode, documents, top_k=10):
    """Run one retrieval configuration and return ranked chunks."""
    query = question
    tokens = tokenize(query)
    by_id = {doc["chunk_id"]: doc for doc in documents}
    if mode == "bm25":
        scores = get_bm25().get_scores(tokens)
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True,)
        return [
            documents[i]
            for i in order[:top_k]
        ]
    if mode == "vector":
        model = get_embedding_model()
        embedding = model.encode(query, normalize_embeddings=True,).tolist()
        collection = get_collection()
        count = min(top_k, collection.count())
        if count == 0:
            return []
        
        result = collection.query(query_embeddings=[embedding], n_results=count, include=["documents", "metadatas", "distances"],)
        return [
            by_id[chunk_id]
            for chunk_id in result["ids"][0]
            if chunk_id in by_id
        ][:top_k]

    if mode not in ("hybrid", "hybrid_reranker"):
        raise ValueError(f"Unknown retrieval mode: {mode}")

    # Hybrid retrieval: combine BM25 and vector ranks using RRF.
    ranked = {}
    scores = get_bm25().get_scores(tokens)
    keyword_order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True,)
    for rank, index in enumerate(keyword_order[:100], start=1):
        doc = documents[index]
        chunk_id = doc["chunk_id"]
        ranked.setdefault(chunk_id, {"doc": doc, "score": 0.0},)
        ranked[chunk_id]["score"] += 1 / (RRF_K + rank)

    # Vector search.
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True,).tolist()
    collection = get_collection()
    count = min(100, len(documents), collection.count())
    if count:
        result = collection.query(query_embeddings=[embedding], n_results=count, include=["documents", "metadatas", "distances"],)
        for rank, chunk_id in enumerate(result["ids"][0], start=1,):
            if chunk_id not in by_id:
                continue

            ranked.setdefault(chunk_id, {"doc": by_id[chunk_id], "score": 0.0,},)
            ranked[chunk_id]["score"] += 1 / (RRF_K + rank)

    # Add keyword-overlap boost before ranking.
    query_terms = set(tokens) - STOP_WORDS
    for item in ranked.values():
        passage_terms = set(tokenize(item["doc"]["text"]))
        overlap = query_terms & passage_terms
        item["score"] += KEYWORD_BOOST * len(overlap)
        item["overlap"] = len(overlap)

    ordered = sorted(ranked.values(),key=lambda item: item["score"],reverse=True,)

    if mode == "hybrid": return [item["doc"] for item in ordered[:top_k]]
    # Hybrid retrieval followed by cross-encoder reranking.
    candidates = ordered[:min(max(top_k * 10, 30), len(ordered))]
    if not candidates: return []
    reranker = get_reranker()
    pairs = [(query, item["doc"]["text"]) for item in candidates]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False,)
    reranked = sorted(zip(candidates, rerank_scores), key=lambda pair: (float(pair[1]) + KEYWORD_BOOST * pair[0]["overlap"]), reverse=True,)
    return [item["doc"] for item, _ in reranked[:top_k]]

def percentile(values, p):
    if not values: return None
    values = sorted(values)
    index = max(0, math.ceil(p * len(values)) - 1,)
    return values[index]

def main():
    questions = load_jsonl(QUESTIONS_PATH)
    documents = load_keyword_corpus()
    answerable = [
        question
        for question in questions
        if question.get("answerable", False)
    ]
    modes = ("bm25", "vector", "hybrid", "hybrid_reranker",)
    rows = []
    summary = {}
    questions_with_urls = sum(
        bool(question.get("supporting_urls"))
        for question in answerable
    )
    print(f"Questions: {len(questions)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Questions with expected URLs: {questions_with_urls}")
    print(f"Indexed chunks: {len(documents)}")
    print("Running retrieval ablations. First run may load models.\n")

    for mode in modes:
        latencies = []
        source_hits = {k: [] for k in TOP_K_VALUES}
        url_coverage = {k: [] for k in TOP_K_VALUES}
        errors = 0
        for index, question in enumerate(answerable, start=1,):
            started = time.perf_counter()
            try:
                results = retrieve_candidates(question["question"], mode, documents, top_k=10,)
                elapsed = (time.perf_counter() - started) * 1000
                latencies.append(elapsed)
                expected_urls = {
                    url.strip().rstrip("/")
                    for url in question.get("supporting_urls", [])
                    if isinstance(url, str) and url.strip()
                }
                for k in TOP_K_VALUES:
                    retrieved_urls = {
                        item.get("metadata", {}).get(
                            "source_url", ""
                        ).strip().rstrip("/")
                        for item in results[:k]
                        if item.get("metadata", {}).get("source_url")
                    }
                    hits = retrieved_urls & expected_urls

                    if expected_urls:
                        source_hits[k].append(1.0 if hits else 0.0)
                        url_coverage[k].append(len(hits) / len(expected_urls))

                rows.append({
                    "id": question["id"],
                    "mode": mode,
                    "latency_ms": round(elapsed, 2),
                    "retrieved_chunk_ids": [item["chunk_id"] for item in results],
                    "retrieved_urls": [item.get("metadata", {}).get("source_url", "") for item in results],
                    "expected_urls": sorted(expected_urls),
                })
                print(f"\r{mode}: {index}/{len(answerable)}", end="", flush=True,)

            except Exception as error:
                errors += 1
                rows.append({"id": question.get("id"), "mode": mode, "error": str(error),})
                print(f"\nERROR {mode} {question.get('id')}: {error}")

        print()

        summary[mode] = {
            "questions_attempted": len(answerable),
            "questions_with_expected_urls": questions_with_urls,
            "errors": errors,
            "expected_source_hit_rate_at_5": (sum(source_hits[5]) / len(source_hits[5]) if source_hits[5] else None),
            "expected_source_hit_rate_at_10": (sum(source_hits[10]) / len(source_hits[10]) if source_hits[10] else None),
            "expected_url_coverage_at_5": (sum(url_coverage[5]) / len(url_coverage[5]) if url_coverage[5] else None),
            "expected_url_coverage_at_10": (sum(url_coverage[10]) / len(url_coverage[10]) if url_coverage[10] else None),
            "search_latency_p50_ms": (round(percentile(latencies, 0.50), 2) if latencies else None),
            "search_latency_p95_ms": (round(percentile(latencies, 0.95), 2) if latencies else None),
        }

    # Save detailed retrieval results.
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save summary metrics.
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False,)

    print("\nRetrieval ablation summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()