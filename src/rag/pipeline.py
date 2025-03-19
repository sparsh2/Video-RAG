"""
Concrete implementation of the RAG pipeline.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.embedding.nomic import NomicTextEmbeddingGenerator, NomicImageEmbeddingGenerator
from src.vector_store.milvus import MilvusStore
from .base import MultimodalRAGPipeline


class NomicRAGPipeline(MultimodalRAGPipeline):
    """RAG pipeline using Nomic embeddings and Milvus."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Nomic RAG pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize embedding generators
        self.text_generator = NomicTextEmbeddingGenerator(config.embedding)
        self.image_generator = NomicImageEmbeddingGenerator(config.embedding)
        
        # Initialize vector store
        self.vector_store = MilvusStore(config.vector_store)
        self.vector_store.connect()
        
        # Initialize language model
        self.model_name = config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        return self.text_generator.generate_embeddings(text)
    
    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get embedding for image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Image embedding
        """
        return self.image_generator.generate_embeddings(image_path)
    
    def _search_vector_store(
        self,
        collection: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store.
        
        Args:
            collection: Collection name
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        return self.vector_store.search(collection, query_embedding, top_k)
    
    def generate_response(
        self,
        query: str,
        retrieved_info: Dict[str, Any],
        **kwargs,
    ) -> str:
        """
        Generate a response based on the query and retrieved information.
        
        Args:
            query: Original query text
            retrieved_info: Retrieved information from the vector store
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Format context from retrieved information
        context = self._format_context(retrieved_info)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()
    
    def add_to_knowledge_base(
        self,
        content: Union[str, List[str], np.ndarray, List[np.ndarray]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        modality: str = "text",
        **kwargs,
    ) -> List[str]:
        """
        Add content to the knowledge base.
        
        Args:
            content: Content to add (text or embeddings)
            metadata: Optional metadata for the content
            modality: Content modality ("text" or "image")
            **kwargs: Additional parameters
            
        Returns:
            List of added content IDs
        """
        # Select collection and generator
        collection = self.text_collection if modality == "text" else self.image_collection
        generator = self.text_generator if modality == "text" else self.image_generator
        
        # Generate embeddings if needed
        if isinstance(content, (str, list)):
            embeddings = generator.generate_embeddings(content)
        else:
            embeddings = content
        
        # Insert into vector store
        vector_ids = self.vector_store.insert(collection, embeddings, metadata)
        
        logger.info(f"Added {len(vector_ids)} {modality} items to knowledge base")
        return vector_ids
    
    def remove_from_knowledge_base(
        self,
        content_ids: List[str],
        **kwargs,
    ) -> None:
        """
        Remove content from the knowledge base.
        
        Args:
            content_ids: List of content IDs to remove
            **kwargs: Additional parameters
        """
        # Remove from both collections
        self.vector_store.delete(self.text_collection, content_ids)
        self.vector_store.delete(self.image_collection, content_ids)
        
        logger.info(f"Removed {len(content_ids)} items from knowledge base")
    
    def _format_context(self, retrieved_info: Dict[str, Any]) -> str:
        """
        Format retrieved information into context.
        
        Args:
            retrieved_info: Retrieved information from the vector store
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for result in retrieved_info["results"]:
            metadata = result["metadata"]
            
            if retrieved_info["modality"] == "text":
                context_parts.append(metadata.get("text", ""))
            else:
                context_parts.append(f"Image: {metadata.get('description', '')}")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the language model.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the user's question:

Context:
{context}

Question: {query}

Answer: [/INST]"""
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "vector_store"):
            self.vector_store.disconnect() 