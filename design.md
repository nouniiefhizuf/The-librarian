# System Design – The Librarian

## 1. Introduction

The Librarian is a Retrieval-Augmented Generation (RAG) system that answers questions using public information from the South Mediterranean University (SMU) website. It retrieves relevant passages and uses OpenAI GPT-4.1-mini to generate answers with source citations.

## 2. Design Objectives

- Answer questions using public SMU website content.
- Retrieve relevant information through semantic and keyword search.
- Generate answers with source citations.
- Provide an accessible API and user interface.
- Evaluate answer quality and system performance.

## 3. System Architecture

The system has two main workflows.

**Knowledge-base preparation:**
1. Collect public SMU website pages.
2. Clean and validate the content.
3. Split documents into chunks.
4. Generate multilingual embeddings.
5. Store vectors in ChromaDB.
6. Prepare the BM25 keyword index.

**Question answering:**
1. Receive a user's question through the API.
2. Retrieve relevant passages using BM25 and vector search.
3. Combine results using Reciprocal Rank Fusion (RRF).
4. Optionally rerank the retrieved passages.
5. Send the question and context to GPT-4.1-mini.
6. Return the generated answer with its sources.

The system can abstain when the retrieved evidence is insufficient.

## 4. Data Collection and Validation

The crawler collects public pages from:

- `www.smu.tn`
- `smu.tn`
- `programs.smu.tn`

Pages are stored in JSONL format with their titles, URLs, and extracted text.

Before processing, the corpus is checked for invalid records, missing information, empty pages, and duplicates.

The latest validation produced **358 valid pages containing 1,038,388 characters**.

## 5. Text Processing and Chunking

Documents are divided into smaller overlapping chunks to improve retrieval and preserve context.

Current configuration:

- Chunk size: 1,200 characters.
- Overlap: 200 characters.
- Minimum page length: 500 characters.

The latest chunking process produced **1,011 unique chunks**, retaining source metadata for citations.

## 6. Embedding and Vector Indexing

The system uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to convert text into numerical vectors.

Its multilingual capabilities support semantic retrieval across English, French, and Modern Standard Arabic.

Vectors are stored in ChromaDB under the collection name `smu_documents`.

## 7. Keyword and Hybrid Retrieval

BM25 provides keyword-based retrieval, which is useful for finding exact terms, program names, and specific information. Its corpus is stored in `storage/bm25_corpus.json`.

The system combines BM25 and vector search using Reciprocal Rank Fusion (RRF), configured with `RRF_K = 60`.

An optional cross-encoder, `BAAI/bge-reranker-v2-m3`, can reorder retrieved passages by relevance. It remains optional because of its high measured latency.

## 8. Answer Generation

The system uses **OpenAI GPT-4.1-mini through the OpenAI API**.

The generation component sends the user's question, retrieved passages, source information, and instructions to the model.

GPT-4.1-mini generates an evidence-based answer with citations. The system can abstain when the retrieved information is insufficient.

The API key is configured through environment variables rather than being hardcoded in the application.

## 9. API and User Interface

The backend uses **FastAPI** to receive questions, validate requests, execute the RAG pipeline, and return answers, sources, and timing information.

The frontend uses **Streamlit**, allowing users to submit questions, read answers, and inspect supporting sources. It communicates with the backend through the configured API URL.

## 10. Evaluation

The project evaluates three main areas:

- **Retrieval evaluation:** Compares BM25, vector search, hybrid retrieval, and reranking using Hit@5, Hit@10, and latency.
- **End-to-end evaluation:** Measures successful executions, answerability decisions, abstentions, expected URL matching, and response time.
- **Manual quality evaluation:** Reviews factual correctness, citation support, and abstention decisions.

Using separate evaluation stages helps identify weaknesses in individual components.

## 11. Design Decisions and Limitations

The multilingual embedding model was selected to support cross-language semantic retrieval. BM25 and vector search were combined to benefit from both keyword matching and semantic similarity.

GPT-4.1-mini was selected for answer generation, while retrieved passages provide evidence for its responses.

Current limitations include:

- Hybrid retrieval has not outperformed BM25 in the recorded evaluation.
- The optional reranker introduces significant latency.
- Factual correctness needs improvement.
- Multilingual performance requires further testing for each language.
- Answer generation depends on OpenAI API availability.

## 12. Conclusion

The Librarian uses a modular architecture that separates data collection, processing, indexing, retrieval, generation, and evaluation.

This design provides a working foundation for multilingual, source-grounded question answering over public SMU website content while allowing individual components to be improved independently.