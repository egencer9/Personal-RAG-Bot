# Scholar-AI

A fully local, private Retrieval-Augmented Generation (RAG) system for document analysis.

## Overview

Scholar-AI is a modular, object-oriented Python application that allows you to chat with your academic and technical documents locally. By leveraging LangChain, Ollama, and ChromaDB, the system ensures 100% privacy with no external API calls for inference or tokenization.

It currently features a functional CLI and is in the process of transitioning to a Streamlit-based user interface.

## Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain
- **LLM Provider:** Ollama (Running Llama 3 locally)
- **Vector Store:** ChromaDB (Persistent local storage)
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Document Loader:** PyPDFLoader

## Architecture

The project follows a modular Object-Oriented Programming pattern with a singleton architecture for database and LLM services to prevent memory leaks during operation.

### Workflow
1. **Ingest:** Load documents via `PyPDFLoader`.
2. **Chunk:** Split documents using `RecursiveCharacterTextSplitter`.
3. **Embed:** Convert chunks into vectors using HuggingFace embeddings.
4. **Store:** Save vectors persistently in ChromaDB (`./vector_db/`).
5. **Query:** Retrieve relevant context for user queries.
6. **Generate:** Produce answers using local Llama 3 via Ollama.

### Key Modules
- `PDFProcessor` (`src/processor.py`): Handles document splitting and chunking.
- `VectorDatabase` (`src/database.py`): Singleton class for managing ChromaDB.
- `LLMService` (`src/llm_service.py`): Singleton class for Ollama connection, prompt templating, and response generation.

## Setup

1. Clone the repository.
2. Ensure you have Python 3.10+ installed.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install [Ollama](https://ollama.com/) and download the Llama 3 model:
   ```bash
   ollama run llama3
   ```
5. Place your PDF documents in the `data/` directory.

## Usage

*Note: The system is currently optimized for local Apple Silicon/CPU hardware and is designed for general academic/technical document domains.*

Run the CLI application to ingest documents and start querying:
```bash
python main.py
```

*(Streamlit UI coming soon!)*
