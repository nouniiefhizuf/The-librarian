# Optimizations and Performance Evaluation

## 1. Introduction

This document summarizes the performance evaluation and optimization of The Librarian, focusing on retrieval quality, answer accuracy, citation support, and response time.

The system uses multilingual embeddings, ChromaDB, BM25, RRF, an optional reranker, and OpenAI GPT-4.1-mini for generation.

## 2. Current Configuration

| Component | Configuration |
|---|---|
| Generation | OpenAI GPT-4.1-mini |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Hybrid retrieval | RRF (K=60) |
| Optional reranker | BAAI/bge-reranker-v2-m3 |
| Chunk size / overlap | 1,200 / 200 characters |

## 3. Corpus and Indexing

The corpus validation process checks invalid records, missing information, empty pages, and duplicates.

Latest results:

| Metric | Result |
|---|---:|
| Valid pages | 358 |
| Total characters | 1,038,388 |
| Unique chunks | 1,011 |
| Vector and BM25 documents | 1,011 each |

## 4. Retrieval Evaluation

Four retrieval methods were compared using 100 questions, including 78 answerable questions with expected source URLs.

| Method | Hit@5 | Hit@10 | P50 latency | P95 latency |
|---|---:|---:|---:|---:|
| BM25 | 58.97% | 70.51% | 2.81 ms | 6.54 ms |
| Vector | 43.59% | 56.41% | 18.41 ms | 25.71 ms |
| Hybrid | 51.28% | 65.38% | 32.78 ms | 39.44 ms |
| Hybrid + reranker | 60.26% | 65.38% | 56.28 s | 80.28 s |

BM25 performed better than the current hybrid configuration. The reranker achieved the highest Hit@5, but its very high latency makes it impractical in its current configuration. Its initialization and inference times should be investigated separately.

## 5. End-to-End Evaluation

| Metric | Result |
|---|---:|
| Questions | 100 |
| Successful runs | 100 |
| Errors | 0 |
| Automated decision accuracy | 96% |
| Correct abstentions | 19/22 |
| Abstention recall | 86.36% |
| Expected URL match rate | 47.44% |
| P50 latency | 1.85 seconds |
| P95 latency | 2.78 seconds |

Expected URL matching measures exact agreement with the expected source URL. It does not necessarily indicate citation correctness because other SMU pages may contain the same supporting information.

## 6. Manual Quality Evaluation

| Metric | Result |
|---|---:|
| Answers reviewed | 78 |
| Factually correct | 48 (61.54%) |
| Citation support | 78/78 (100%) |
| Manual abstention accuracy | 80% |
| Remaining reviews | 0 |

Although all reviewed answerable responses had supporting citations, only 61.54% were judged factually correct. This shows that relevant citations do not automatically guarantee accurate answers.

Manual abstention accuracy differs from automated decision accuracy, so the two results should be interpreted separately.

## 7. Main Optimization Priorities

1. **Factual correctness:** Improve generation instructions and ensure answers remain grounded in retrieved evidence.
2. **Retrieval quality:** Tune BM25, vector search, and RRF to improve relevant passage selection.
3. **Reranker performance:** Investigate initialization overhead and reduce the number of candidates reranked.
4. **Abstention:** Improve evidence sufficiency checks and review missed abstentions.
5. **Multilingual support:** Evaluate English, French, and Modern Standard Arabic separately, including Arabic normalization and cross-language retrieval.

## 8. Recommended Configuration

- Keep GPT-4.1-mini for generation.
- Keep the multilingual embedding model.
- Continue evaluating BM25 and hybrid retrieval.
- Keep the reranker optional until its latency improves.
- Prioritize factual correctness and manual evaluation.

## 9. Conclusion

The Librarian successfully completed its end-to-end evaluation without execution errors and achieved 100% citation support in the manually reviewed sample. However, factual correctness and retrieval effectiveness still need improvement.

The next optimization phase should focus on selecting better evidence and helping GPT-4.1-mini interpret it accurately while maintaining reasonable response times.