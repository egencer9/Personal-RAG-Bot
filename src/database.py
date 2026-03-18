import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDatabase:
    _instance = None

    def __new__(cls, persist_directory="./vector_db"):
        if cls._instance is None:
            cls._instance = super(VectorDatabase, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_directory="./vector_db"):
        if self._initialized:
            return
            
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None
        self._init_db()
        self._initialized = True

    def _init_db(self):
        """Initializes the Chroma database."""
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
            
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

    def add_documents(self, documents):
        """Adds chunked documents to the vector database."""
        if not documents:
            return
            
        print(f"Adding {len(documents)} documents to the vector database...")
        self.db.add_documents(documents)
        print("Documents successfully added to the database.")

    def search(self, query, top_k=3):
        """Performs similarity search against the vector database."""
        if self.db is None:
            return []
            
        print(f"Searching for: '{query}'")
        docs = self.db.similarity_search(query, k=top_k)
        return docs
