"""
PDF Document Ingestion Module for Suspicious Account Activity and Billing Chatbot.

This module handles the ingestion of PDF documents, extraction of text and images,
and creation of vector embeddings for semantic search and retrieval.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import tiktoken
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentIngester:
    """
    Handles PDF document ingestion, processing, and vector storage creation.
    
    This class processes PDF documents for the chatbot, including text extraction,
    chunking, embedding generation, and vector storage.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    ):
        """
        Initialize the DocumentIngester.
        
        Args:
            chunk_size: The size of text chunks for processing
            chunk_overlap: The overlap between consecutive chunks
            embedding_model: The name of the embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_counter,
        )
        
    def _token_counter(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoder.encode(text))
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Extract images from PDF and save them to the output directory.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List[str]: Paths to the extracted image files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_paths = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Generate filename based on PDF name, page number, and image index
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                # Save the image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                    
                image_paths.append(image_path)
                
        return image_paths
    
    def process_pdf(self, pdf_path: str, images_dir: str = "static/images") -> List[Document]:
        """
        Process a PDF file into LangChain documents with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            images_dir: Directory to save extracted images
            
        Returns:
            List[Document]: Processed documents with metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract images
        image_paths = self.extract_images_from_pdf(pdf_path, images_dir)
        logger.info(f"Extracted {len(image_paths)} images from {pdf_path}")
        
        # Load text using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = pdf_path
            doc.metadata["pdf_name"] = os.path.basename(pdf_path)
            
            # Add available images for this page
            page_num = doc.metadata.get("page", 0)
            doc.metadata["images"] = [
                img_path for img_path in image_paths 
                if f"page{page_num+1}" in img_path
            ]
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        return split_docs
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        persist_directory: str
    ) -> FAISS:
        """
        Create and persist a FAISS vector store from documents.
        
        Args:
            documents: List of processed documents
            persist_directory: Directory to save the vector store
            
        Returns:
            FAISS: The created vector store
        """
        logger.info(f"Creating vector store in {persist_directory}")
        
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Create directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        # Save vector store
        vector_store.save_local(persist_directory)
        logger.info(f"Vector store saved to {persist_directory}")
        
        return vector_store
    
    def load_vector_store(self, persist_directory: str) -> Optional[FAISS]:
        """
        Load a FAISS vector store from disk.
        
        Args:
            persist_directory: Directory containing the vector store
            
        Returns:
            Optional[FAISS]: The loaded vector store or None if it doesn't exist
        """
        if not os.path.exists(persist_directory):
            logger.warning(f"Vector store directory {persist_directory} does not exist")
            return None
            
        logger.info(f"Loading vector store from {persist_directory}")
        return FAISS.load_local(persist_directory, self.embeddings)
    
    def process_pdf_directory(
        self, 
        pdf_directory: str, 
        images_dir: str = "static/images",
        vector_store_dir: str = "models/vector_store"
    ) -> FAISS:
        """
        Process all PDFs in a directory and create a combined vector store.
        
        Args:
            pdf_directory: Directory containing PDF files
            images_dir: Directory to save extracted images
            vector_store_dir: Directory to save the vector store
            
        Returns:
            FAISS: The created vector store
        """
        all_documents = []
        
        # Process each PDF in the directory
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, filename)
                documents = self.process_pdf(pdf_path, images_dir)
                all_documents.extend(documents)
        
        logger.info(f"Processed {len(all_documents)} total chunks from all PDFs")
        
        # Create and save vector store
        return self.create_vector_store(all_documents, vector_store_dir)


if __name__ == "__main__":
    # Example usage
    ingester = DocumentIngester()
    
    # Process PDFs
    pdf_dir = "data"
    ingester.process_pdf_directory(
        pdf_directory=pdf_dir,
        images_dir="static/images",
        vector_store_dir="models/vector_store"
    )
    
    logger.info("Document ingestion complete!")