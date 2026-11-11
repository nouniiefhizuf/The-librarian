# Project Postmortem – The Librarian

## 1. Introduction

This document summarizes the development, evaluation, successes, challenges, and possible improvements of The Librarian, a RAG system built using public SMU website content.

## 2. Project Objectives

The project aimed to build a system that collects SMU website information, retrieves relevant passages, generates answers using OpenAI GPT-4.1-mini, provides citations, and supports English, French, and Modern Standard Arabic through an API and Streamlit interface.

## 3. What Went Well

- **Corpus collection:** Successfully collected 358 valid pages containing 1,038,388 characters.
- **Chunking and indexing:** Produced 1,011 unique chunks and successfully built both vector and BM25 indexes.
- **Application:** Successfully integrated ingestion, retrieval, GPT-4.1-mini generation, FastAPI, and Streamlit.
- **Execution:** Completed all 100 end-to-end evaluation runs without errors.
- **Citation support:** All 78 manually reviewed answerable responses had supporting citations.

## 4. What Did Not Go Well

- **Factual correctness:** Only 48 of 78 reviewed answers were correct (61.54%).
- **Retrieval:** BM25 outperformed the current hybrid configuration in Hit@5 and Hit@10.
- **Reranker latency:** The reranker achieved 60.26% Hit@5 but required approximately 56.28 seconds at P50.
- **Abstention:** Automated evaluation recorded 19/22 correct abstentions, while manual abstention accuracy was 80%.
- **URL matching:** Exact expected-source matching reached 47.44%, although alternative SMU pages may also support an answer.

## 5. Evaluation Summary

| Metric | Result |
|---|---:|
| End-to-end questions | 100 |
| Successful runs | 100 |
| Errors | 0 |
| Automated decision accuracy | 96% |
| Correct automated abstentions | 19/22 |
| Expected URL match rate | 47.44% |
| Median latency | 1.85 seconds |
| P95 latency | 2.78 seconds |
| Manual factual correctness | 61.54% |
| Manual citation support | 100% |
| Manual abstention accuracy | 80% |

## 6. Possible Causes of Problems

The following are potential causes that require further investigation:

- **Incomplete retrieval:** Retrieved passages may not contain all the facts needed to answer a question.
- **Overlapping content:** Similar information across SMU pages may affect retrieval ranking.
- **Generation instructions:** The prompt may need stronger evidence and accuracy requirements.
- **Retrieval configuration:** RRF parameters may not combine BM25 and vector results effectively.
- **Reranker overhead:** Model initialization or inference may explain its high latency.

## 7. Lessons Learned

1. A larger corpus does not guarantee accurate answers.
2. Retrieval and generation quality must be evaluated separately.
3. Citation support and factual correctness are different metrics.
4. Hybrid retrieval does not automatically outperform individual methods.
5. Performance improvements must justify their latency costs.
6. Manual evaluation is important for identifying quality problems.
7. Multilingual systems require language-specific testing.

## 8. Corrective Actions

| Priority | Action |
|---|---|
| High | Improve generation instructions and factual accuracy. |
| High | Review incorrect answers and missed abstentions. |
| High | Tune BM25 and hybrid retrieval. |
| Medium | Investigate and reduce reranker latency. |
| Medium | Expand English, French, and Arabic evaluation. |
| Medium | Improve citation evaluation. |

## 9. Final Assessment

The Librarian successfully delivers a working RAG application with public SMU content, hybrid retrieval, GPT-4.1-mini generation, citations, an API, and a user interface.

Although all 100 end-to-end tests completed successfully, factual correctness and retrieval efficiency remain the main challenges.

The project is therefore a functional prototype with measurable results and clear opportunities for improvement, rather than a fully optimized production system.