# IIT Patna Academic Assistant

A Retrieval-Augmented Generation (RAG) based academic assistant for IIT Patna. The system retrieves relevant information from an indexed collection of IIT Patna academic webpages and PDF documents and uses a small language model to generate grounded answers.

## Overview

```text
User Query
    ↓
BGE Sentence Embedding
    ↓
FAISS Semantic Search
    ↓
Top-k Relevant Document Chunks
    ↓
Context Augmentation
    ↓
TinyLlama-1.1B
    ↓
Grounded Answer
```

The assistant is designed to answer questions related to academic calendars, syllabi, registration, semester schedules, admission-related information, and other information present in the indexed academic database.

## Tech Stack

- **Python**
- **Streamlit** — interactive chatbot interface
- **Sentence Transformers** — query/document embeddings
- **BAAI/bge-base-en-v1.5** — embedding model
- **FAISS** — vector similarity search
- **Hugging Face Transformers** — model inference
- **TinyLlama-1.1B-Chat-v1.0** — Small Language Model for answer generation
- **Jupyter/Google Colab** — data ingestion and experimentation

## Project Structure

```text
IIT_Patna_Chatbot/
│
├── app.py
├── index/
│   ├── index.faiss
│   ├── chunks.npy
│   └── metadata.npy
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   └── 02_rag_inference.ipynb
│
├── docs/
│   └── IIT_Patna_Academic_Assistant.pdf
│
├── images/
│   └── demo.png
│
├── requirements.txt
└── README.md
```

## RAG Pipeline

### 1. Data Ingestion
Official IIT Patna webpages and PDF documents are collected and processed into text.

### 2. Chunking
Extracted text is divided into smaller chunks suitable for semantic retrieval.

### 3. Embedding
Document chunks are converted into dense vector representations using `BAAI/bge-base-en-v1.5`.

### 4. Vector Indexing
The embeddings are stored in a FAISS index. The generated artifacts are:

- `index.faiss`
- `chunks.npy`
- `metadata.npy`

### 5. Retrieval
A user query is embedded using the same embedding model and FAISS retrieves the top relevant chunks.

### 6. Generation
The retrieved context is passed to TinyLlama-1.1B-Chat-v1.0. The prompt instructs the model to answer only from the supplied context and avoid guessing when information is unavailable.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/chinmaycodes7/IIT_Patna_Chatbot.git
cd IIT_Patna_Chatbot
```

### 2. Install dependencies

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The FAISS index and NumPy artifacts should be present in the `index/` directory. On the first run, the required Hugging Face models will be downloaded automatically.

## Example Query

> What is the eligibility for M.Tech admission at IIT Patna?

The system retrieves relevant academic document chunks and generates an answer based on the retrieved context.

## Notebooks

### `01_data_ingestion.ipynb`

Covers:
- Website crawling
- PDF extraction
- Text processing
- Chunking
- Embedding generation
- FAISS index construction

### `02_rag_inference.ipynb`

Covers:
- Loading the FAISS index
- Query embedding
- Semantic retrieval
- Context construction
- TinyLlama-based answer generation

## Design Choices

### Why RAG?
Academic information is distributed across multiple webpages and documents. RAG allows the system to retrieve relevant source information rather than relying entirely on knowledge stored in the language model.

### Why a Small Language Model?
TinyLlama provides a lightweight generation component with lower compute and memory requirements than larger language models, making it suitable for experimentation and resource-constrained environments.

### Why FAISS?
FAISS provides efficient vector similarity search and allows retrieved document chunks to be supplied as external context to the language model.

## Limitations

- Answer quality depends on the quality and coverage of the indexed documents.
- The system can only answer questions supported by the indexed knowledge base.
- TinyLlama is a relatively small language model, so generation quality is limited compared with larger models.
- The FAISS index needs to be rebuilt when the underlying document collection is substantially updated.
- The current chat interface does not display source citations for retrieved chunks.

## Future Improvements

- Automatic periodic crawling and index updates
- Source/document citations in generated answers
- Improved chunking and metadata-aware retrieval
- Hybrid keyword + semantic retrieval
- Reranking retrieved passages
- Evaluation using a curated IIT Patna QA benchmark
- Comparison of different embedding and language models
- Improved conversational memory and multi-turn question handling

## Project Presentation

The project presentation is available in the `docs/` directory.

## Disclaimer

This project is an academic/research prototype. Answers are generated from the indexed academic database and should be verified against the latest official IIT Patna sources for time-sensitive or high-stakes information.
