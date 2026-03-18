import os
import argparse
from src.processor import PDFProcessor
from src.database import VectorDatabase
from src.llm_service import LLMService

def ingest():
    """Ingests all PDFs from the data directory into ChromaDB."""
    processor = PDFProcessor()
    documents = processor.load_and_chunk_all()
    
    if documents:
        db = VectorDatabase()
        db.add_documents(documents)
    else:
        print("No documents found to process. Please place PDFs in the 'data' directory.")

def chat(query):
    """Executes the RAG loop for a given query."""
    db = VectorDatabase()
    llm = LLMService()
    
    docs = db.search(query)
    if docs:
        print("\n" + "="*50)
        print("Retrieved Context:")
        print("="*50)
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:")
            print(doc.page_content)
            print("-" * 20)
        
        print("\n" + "="*50)
        print("Generated Response:")
        print("="*50)
        llm.generate_response(query, docs)
        print("="*50 + "\n")
    else:
        print("No relevant context found in the database. Ensure documents are ingested first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scholar-AI (Local RAG System)")
    parser.add_argument(
        "--ingest", 
        action="store_true", 
        help="Process PDFs from the data directory and update the vector database."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Ask a question against the ingested documents."
    )
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest()
    elif args.query:
        chat(args.query)
    else:
        parser.print_help()