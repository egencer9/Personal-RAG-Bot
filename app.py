import os
import streamlit as st
from src.processor import PDFProcessor
from src.database import VectorDatabase
from src.llm_service import LLMService

# --- Page Configuration ---
st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# --- Constants & Directories ---
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- Singleton Initializations ---
# These will be initialized only once and reused across the app
@st.cache_resource
def get_llm_service():
    return LLMService()

@st.cache_resource
def get_vector_db():
    return VectorDatabase()

llm_service = get_llm_service()
vector_db = get_vector_db()

# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header("📚 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents here",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            # Save uploaded files to the data directory
            saved_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)
            
            with st.spinner("Processing documents... This may take a moment."):
                processor = PDFProcessor(data_dir=DATA_DIR)
                documents = processor.load_and_chunk_all()
                if documents:
                    vector_db.add_documents(documents)
                    st.success("✅ Documents processed and stored successfully!")
                else:
                    st.warning("No new documents were found to process.")
                    
            # Clean up files after processing so they aren't processed again on next run
            for path in saved_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    pass
        else:
            st.warning("Please upload at least one PDF file.")

# --- Main Chat Interface ---
st.title("💬 Personal Assistant")
st.caption("A local-first RAG assistant powered by Llama 3 and Streamlit")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your documents today?"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Perform RAG
            retrieved_docs = vector_db.search(prompt)
            
            # (Optional) Display retrieved context
            if retrieved_docs:
                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Document {i+1}:**")
                        st.caption(doc.page_content)
                        st.divider()
            else:
                st.warning("No relevant context found in the database. I will try to answer based on my general knowledge or inform you that I do not know.")
            
            # We use st.write_stream to natively stream the Langchain generator
            response_generator = llm_service.generate_response_stream(prompt, retrieved_docs)
            full_response = st.write_stream(response_generator)
            
        # Add the full response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})