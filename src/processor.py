import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def load_and_chunk_all(self):
        """Loads all PDFs from the data directory and returns chunked documents."""
        all_chunks = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created directory {self.data_dir}. Please place PDF files here.")
            return all_chunks

        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        
        for file_path in pdf_files:
            print(f"Processing {file_path}...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"Generated {len(chunks)} chunks from {os.path.basename(file_path)}")
            
        return all_chunks
