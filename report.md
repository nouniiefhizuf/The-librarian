# The Librarian: A Multilingual RAG System for SMU

**Course:** CS496 – AI Engineering  
**Project:** Project 2 – The Librarian  
**Institution:** Mediterranean Institute of Technology  
**Term:** Fall 2026

## Abstract

The Librarian is a multilingual Retrieval-Augmented Generation (RAG) system that answers questions using public South Mediterranean University (SMU) website content.

It combines multilingual embeddings, ChromaDB, BM25, and Reciprocal Rank Fusion (RRF) to retrieve relevant information. OpenAI GPT-4.1-mini then generates answers with source citations.

The system includes website ingestion, indexing, retrieval, answer generation, a FastAPI backend, a Streamlit interface, and evaluation tools. Its latest corpus contains 358 pages and 1,011 unique chunks. Evaluation achieved 96% automated decision accuracy, while manual review found 61.54% factual correctness and 100% citation support among reviewed answers.

## 1. Introduction

University information is often spread across multiple website pages, making specific answers difficult to find.

The Librarian addresses this problem through a question-answering system that retrieves relevant SMU content before generating responses. This helps ground answers in available website information rather than relying only on the model's internal knowledge.

## 2. Problem and Objectives

The project aims to make public university information easier to access through natural-language questions.

Its objectives are to:

- Collect and validate public SMU website pages.
- Build a searchable knowledge base.
- Support semantic and keyword retrieval.
- Combine retrieval methods using RRF.
- Generate answers with GPT-4.1-mini.
- Provide source citations and abstain when evidence is insufficient.
- Support English, French, and Modern Standard Arabic.
- Evaluate answer quality, retrieval, citations, and latency.

## 3. Scope

The corpus contains publicly accessible pages from:

- `www.smu.tn`
- `smu.tn`
- `programs.smu.tn`

The project does not use private student records or confidential university information. Answers are limited to the indexed content, which may become outdated when the website changes.

## 4. System Architecture

The system has two main workflows.

**Knowledge-base preparation:**
1. Crawl public SMU pages.
2. Extract and clean text.
3. Validate the corpus.
4. Split documents into chunks.
5. Generate multilingual embeddings.
6. Build ChromaDB and BM25 indexes.

**Question answering:**
1. Receive a user question through Streamlit.
2. Send it to the FastAPI backend.
3. Retrieve relevant passages using BM25 and vector search.
4. Combine results using RRF.
5. Optionally rerank the results.
6. Send the question and retrieved context to GPT-4.1-mini.
7. Return the answer and source citations, or abstain if evidence is insufficient.

## 5. Technologies and Implementation

| Component | Technology |
|---|---|
| Programming language | Python |
| Generation model | OpenAI GPT-4.1-mini |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Result combination | RRF (K=60) |
| Optional reranker | BAAI/bge-reranker-v2-m3 |
| Backend | FastAPI |
| Interface | Streamlit |

GPT-4.1-mini is accessed through the OpenAI API.

The crawler collects public SMU pages and stores their text and metadata. After validation, documents are split into chunks using a size of 1,200 characters, an overlap of 200 characters, and a minimum page length of 500 characters.

The multilingual embedding model creates vector representations stored in the `smu_documents` ChromaDB collection. BM25 uses a separate keyword corpus. The retrieval component supports BM25, vector search, hybrid search, and optional reranking.

The generation component uses retrieved passages as context and returns answers with source information.

## 6. Evaluation Methodology

Three evaluation stages were used:

- **Retrieval evaluation:** compares BM25, vector, hybrid, and reranked retrieval using Hit@5, Hit@10, and latency.
- **End-to-end evaluation:** measures successful runs, answerability decisions, abstentions, source URL matching, and response time.
- **Manual quality evaluation:** reviews factual correctness, citation support, and abstention decisions.

Manual review is important because exact URL matching alone cannot determine whether a citation genuinely supports an answer.

## 7. Results

### 7.1 Corpus and indexing

| Metric | Result |
|---|---:|
| Valid pages | 358 |
| Total characters | 1,038,388 |
| Unique chunks | 1,011 |
| ChromaDB documents | 1,011 |
| BM25 documents | 1,011 |

### 7.2 Retrieval comparison

| Method | Hit@5 | Hit@10 | P50 latency |
|---|---:|---:|---:|
| BM25 | 58.97% | 70.51% | 2.81 ms |
| Vector | 43.59% | 56.41% | 18.41 ms |
| Hybrid | 51.28% | 65.38% | 32.78 ms |
| Hybrid + reranker | 60.26% | 65.38% | 56.28 s |

BM25 outperformed the current hybrid configuration in Hit@5 and Hit@10. The reranker achieved the highest Hit@5 but introduced substantial latency.

### 7.3 End-to-end evaluation

| Metric | Result |
|---|---:|
| Questions | 100 |
| Successful runs | 100 |
| Execution errors | 0 |
| Automated decision accuracy | 96% |
| Correct automated abstentions | 19/22 |
| Abstention recall | 86.36% |
| Expected URL match rate | 47.44% |
| P50 latency | 1.85 s |
| P95 latency | 2.78 s |

The expected URL match rate measures exact matches with expected URLs; it does not necessarily indicate whether other citations are unsupported.

### 7.4 Manual quality evaluation

| Metric | Result |
|---|---:|
| Answerable answers reviewed | 78 |
| Factually correct | 48 (61.54%) |
| Citation support | 78/78 (100%) |
| Manual abstention accuracy | 80/100 |
| Remaining reviews | 0 |

The results show that citation support does not automatically guarantee factual correctness. Automated and manual abstention results are separate measurements and should not be combined.

## 8. Discussion and Limitations

The system successfully completed all 100 end-to-end evaluation runs and produced citations judged supportive in the reviewed sample.

However, factual correctness remains its main weakness. The current hybrid retrieval configuration also performs worse than BM25 on the recorded expected-source hit rates. The reranker provides a small retrieval improvement at a very high measured latency.

Other limitations include:

- Multilingual quality needs separate evaluation for each language.
- Exact URL matching is an imperfect citation metric.
- The system depends on OpenAI API availability.
- The knowledge base requires updates when SMU content changes.

## 9. Future Work

Future improvements should focus on:

- Improving answer accuracy and generation instructions.
- Tuning BM25, vector retrieval, and RRF parameters.
- Investigating reranker initialization and inference latency.
- Improving abstention and evidence sufficiency checks.
- Expanding English, French, and Arabic evaluation.
- Strengthening citation validation.
- Automating regular corpus updates.

## 10. Conclusion

The Librarian demonstrates a working multilingual RAG application over public SMU website content. It integrates ingestion, indexing, retrieval, GPT-4.1-mini generation, citations, an API, and a user interface.

Evaluation confirms that the system works end to end while identifying factual correctness, retrieval quality, abstention behavior, and latency as key areas for improvement.