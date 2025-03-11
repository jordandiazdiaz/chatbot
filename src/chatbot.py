"""
Chatbot Core Module for Suspicious Account Activity and Billing Support.

This module implements the core chatbot functionality, including
context-aware responses, security measures, and query handling.
"""

import os
import re
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SupportChatbot:
    """
    Implements a context-aware support chatbot for suspicious account activity and billing inquiries.
    
    This chatbot uses semantic search to find relevant information in the provided PDFs
    and responds to user queries within its domain of knowledge.
    """
    
    def __init__(
        self,
        vector_store_path: str = "models/vector_store",
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        completion_model: str = os.getenv("COMPLETION_MODEL", "gpt-4-turbo"),
        temperature: float = float(os.getenv("TEMPERATURE", 0.1)),
        max_tokens: int = int(os.getenv("MAX_TOKENS", 512)),
    ):
        """
        Initialize the Support Chatbot.
        
        Args:
            vector_store_path: Path to the FAISS vector store
            embedding_model: Name of the embedding model to use
            completion_model: Name of the completion model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in generated responses
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Load vector store
        self._load_vector_store()
        
        # Initialize memory and chain
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create chain
        self._create_chain()
        
        # Define restricted keywords that indicate out-of-scope queries
        self.restricted_keywords = [
            "password", "contraseña", "login credentials", "credit card", 
            "tarjeta", "social security", "seguro social", "address", 
            "dirección", "phone number", "teléfono", "email personal",
            "technical support", "soporte técnico", "device", "dispositivo",
            "installation", "instalación", "content", "contenido", 
            "programming", "programación", "shows", "películas", "movies"
        ]
        
    def _load_vector_store(self):
        """Load the FAISS vector store from disk."""
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
            
        logger.info(f"Loading vector store from {self.vector_store_path}")
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
    
    def _create_chain(self):
        """Create the conversational retrieval chain."""
        # Define custom prompt templates
        qa_prompt = PromptTemplate(
            template="""You are a support assistant for a streaming platform, helping with account activity and billing issues.
You ONLY answer questions related to suspicious account activity and billing procedures.
You DO NOT answer questions about anything else.

If a question is outside of your knowledge about account activity and billing procedures, 
respond with: "I'm sorry, but I can only assist with questions related to account activity and billing procedures."

Use the following context to answer the question:
{context}

If the context doesn't provide enough information to answer the question confidently, say:
"I don't have enough information about that in my knowledge base. Please contact our support team for assistance."

Always include relevant images or diagrams if they are mentioned in the context, using the format:
[Image: image_filename]

Question: {question}
Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create LLM
        llm = ChatOpenAI(
            model=self.completion_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Create chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5 chunks
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
    
    def _sanitize_input(self, query: str) -> str:
        """
        Sanitize user input to prevent prompt injection attacks.
        
        Args:
            query: User query string
            
        Returns:
            str: Sanitized query
        """
        # Remove potentially harmful characters and patterns
        sanitized = query.strip()
        
        # Remove any attempt to inject system instructions
        sanitized = re.sub(r'(system:|assistant:|user:|<|>|\[|\]|\{|\})', '', sanitized)
        
        # Limit query length
        if len(sanitized) > 500:
            sanitized = sanitized[:500]
            
        return sanitized
    
    def _is_within_scope(self, query: str) -> bool:
        """
        Check if a query is within the chatbot's scope of knowledge.
        
        Args:
            query: User query string
            
        Returns:
            bool: True if query is within scope, False otherwise
        """
        # Check for restricted keywords
        query_lower = query.lower()
        for keyword in self.restricted_keywords:
            if keyword.lower() in query_lower:
                return False
                
        # Perform semantic search to see if we have relevant content
        docs = self.vector_store.similarity_search(query, k=3)
        
        # If we didn't find any relevant documents, query may be out of scope
        if not docs:
            return False
            
        # Calculate average relevance score (if possible)
        # If this is not available directly, we can skip this check
        
        return True
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return a response with metadata.
        
        Args:
            query: User query string
            
        Returns:
            Dict[str, Any]: Response with answer, sources, and metadata
        """
        # Sanitize input
        sanitized_query = self._sanitize_input(query)
        
        # Check if query is within scope
        if not self._is_within_scope(sanitized_query):
            return {
                "answer": "I'm sorry, but I can only assist with questions related to account activity and billing procedures.",
                "sources": [],
                "images": [],
                "token_usage": None,
                "is_fallback": True
            }
        
        # Track token usage
        with get_openai_callback() as cb:
            # Get response from chain
            response = self.chain.invoke({"question": sanitized_query})
            
            # Extract answer and source documents
            answer = response.get("answer", "")
            source_documents = response.get("source_documents", [])
            
            # Extract sources and images
            sources = []
            images = []
            
            for doc in source_documents:
                # Add source
                source = doc.metadata.get("source", "")
                if source and source not in sources:
                    sources.append(source)
                
                # Add images
                doc_images = doc.metadata.get("images", [])
                for img in doc_images:
                    if img and img not in images:
                        images.append(img)
            
            # Process answer for image references
            img_pattern = r'\[Image: ([^\]]+)\]'
            image_references = re.findall(img_pattern, answer)
            
            # Add any mentioned images that might not be in metadata
            for img_ref in image_references:
                for img_path in images:
                    if img_ref in img_path and img_path not in images:
                        images.append(img_path)
            
        return {
            "answer": answer,
            "sources": sources,
            "images": images,
            "token_usage": {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "cost": cb.total_cost
            },
            "is_fallback": False
        }
    
    def reset_conversation(self):
        """Reset the conversation memory."""
        self.memory.clear()


# Test code for direct module execution
if __name__ == "__main__":
    chatbot = SupportChatbot()
    
    # Test with sample queries
    test_queries = [
        "What should I do if I see suspicious activity on my account?",
        "How can I report unauthorized charges?",
        "Tell me about the latest movies on your platform",  # Out of scope
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.get_answer(query)
        print(f"Answer: {response['answer']}")
        print(f"Sources: {response['sources']}")
        print(f"Images: {response['images']}")
        print(f"Token Usage: {response['token_usage']}")
        print(f"Is Fallback: {response['is_fallback']}")