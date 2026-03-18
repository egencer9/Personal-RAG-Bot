from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

class LLMService:
    _instance = None

    def __new__(cls, model_name="llama3"):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name="llama3"):
        if self._initialized:
            return
            
        self.model_name = model_name
        self.llm = Ollama(model=self.model_name)
        self.prompt = self._create_prompt()
        self._initialized = True

    def _create_prompt(self):
        """Creates the LangChain prompt template."""
        template = """You are a helpful assistant. Use only the provided context to answer the question. If the answer is not in the context, state that you do not know. Do not hallucinate.

Context: {context}

Question: {question}

Answer:"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def generate_response(self, query, retrieved_docs):
        """Generates a response using Ollama and the retrieved context."""
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        chain = self.prompt | self.llm
        
        # Use stream for better user experience
        print("Generating response...\n")
        response = ""
        for chunk in chain.stream({"context": context, "question": query}):
            print(chunk, end="", flush=True)
            response += chunk
        print()

        return response
