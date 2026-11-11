# The Librarian

A multilingual Retrieval-Augmented Generation (RAG) system that answers questions using public South Mediterranean University (SMU) website content. It uses GPT-4.1-mini to generate answers supported by source citations.

## 1. Project Overview

Developed for the CS496 – AI Engineering course, The Librarian retrieves relevant information from an indexed collection of SMU pages before generating answers.

It is designed to support English, French, and Modern Standard Arabic.

## 2. Main Features

- Collects and processes public SMU website pages.
- Splits documents into searchable chunks.
- Generates multilingual embeddings.
- Uses ChromaDB for vector search and BM25 for keyword search.
- Combines results using Reciprocal Rank Fusion (RRF).
- Supports optional cross-encoder reranking.
- Generates answers using OpenAI GPT-4.1-mini.
- Provides source citations and abstains when evidence is insufficient.
- Includes a FastAPI backend, Streamlit interface, and evaluation tools.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Generation | OpenAI GPT-4.1-mini |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector database | ChromaDB |
| Keyword search | BM25 |
| Hybrid retrieval | Reciprocal Rank Fusion |
| Optional reranker | BAAI/bge-reranker-v2-m3 |
| Backend | FastAPI |
| Interface | Streamlit |

## 4. Project Structure

- `app/`: FastAPI backend (`main.py`) and Streamlit interface (`ui.py`).
- `data/`: Collected SMU pages and processed chunks.
- `src/`: Website ingestion, validation, chunking, indexing, retrieval, and generation.
- `eval/`: Evaluation questions, scripts, manual reviews, and results.
- `storage/`: ChromaDB index and BM25 corpus.
- `scripts/`: Setup and local launcher scripts.
- `.env` and `.env.example`: Environment configuration.

## 5. Requirements

- Python 3.10 or a compatible version.
- OpenAI API key with GPT-4.1-mini access.
- Internet access for website ingestion and API requests.
- Local storage for documents and indexes.

## 6. Installation

Open PowerShell in the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies using the project's dependency file. For example:

```powershell
pip install -r requirements.txt
```

## 7. API Configuration

Add your OpenAI API key to the root `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

Keep `.env` private and excluded from Git. Use `.env.example` to document required variables without exposing secrets.

## 8. Building the Knowledge Base

The knowledge-base workflow consists of four steps:

```powershell
python -m src.ingest_web
python -m src.check_corpus
python -m src.chunking
python -m src.indexing
```

These scripts collect SMU pages, validate the corpus, create chunks, and build the retrieval indexes.

## 9. Running the Application

Start the API:

```powershell
uvicorn app.main:app --reload
```

In another terminal, start the interface:

```powershell
streamlit run app/ui.py
```

The Streamlit interface communicates with the FastAPI backend.

## 10. Evaluation

Run the evaluation scripts:

```powershell
python -m eval.run_eval
python -m eval.retrieval_eval
python eval/quality_eval.py
python eval/audit_sources.py
```

They measure retrieval effectiveness, answer quality, abstention behavior, citation support, and latency.

## 11. Evaluation Snapshot

**Corpus**
- 358 valid pages.
- 1,038,388 characters.
- 1,011 unique chunks.
- 1,011 vector and BM25 documents.

**End-to-end evaluation**
- 100 questions and 100 successful runs.
- 0 execution errors.
- 96% automated decision accuracy.
- 19/22 correct abstentions.
- 47.44% expected URL match rate.
- P50 latency: 1.85 seconds.
- P95 latency: 2.78 seconds.

**Manual quality evaluation**
- Factual correctness: 48/78 (61.54%).
- Citation support: 78/78 (100%).
- Manual abstention accuracy: 80/100.
- Remaining manual reviews: 0.

These figures describe the recorded evaluation run, not guaranteed future performance.

## 12. Limitations and Future Improvements

Current limitations include:
- Factual correctness needs improvement.
- Some answers are incomplete or incorrect.
- The reranker introduces significant latency.
- Multilingual answer quality requires further testing.
- The system depends on OpenAI API availability.
- Website changes can make the indexed knowledge base outdated.

Future work should focus on improving answer accuracy, tuning retrieval, evaluating all three languages, reducing latency, and updating the corpus regularly.

## 13. Conclusion

The Librarian provides a working RAG application over public SMU website content. It integrates multilingual embeddings, vector and keyword retrieval, GPT-4.1-mini generation, citations, an API, and a user interface.

Evaluation confirms that the system works end to end while highlighting factual correctness and retrieval efficiency as its main areas for improvement.