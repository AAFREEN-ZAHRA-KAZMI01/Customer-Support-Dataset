from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.embeddings import EmbeddingGenerator
from src.qdrant_client import QdrantVectorStore
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import random

load_dotenv()

class RAGPipeline:
    def __init__(self, collection_name: str = "ecommerce_faq"):
        """Initialize the RAG pipeline with vector store and LLM"""
        self.embedder = EmbeddingGenerator()
        self.vector_db = QdrantVectorStore(collection_name=collection_name)
        
        # Initialize LLM with fallback
        try:
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                request_timeout=30
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

        # Greeting responses
        self.greeting_responses = [
            "🛍️ Welcome to [Brand] Support! How can I help with:\n- Orders\n- Returns\n- Payments\n- Account issues",
            "👋 Hello! I'm your shopping assistant. What can I help you with today?",
            "🌟 Welcome back! Need help with your recent order or account?"
        ]
        
        # Fallback responses
        self.fallback_responses = [
            "Let me check... I couldn't find exact information, but try:\n1. Our help center: [link]\n2. Email support@example.com",
            "I'm still learning about this. For immediate help:\n• Visit our FAQ\n• Contact live chat",
            "This topic isn't in my knowledge base yet. Our team can help at support@example.com"
        ]

        # Optimized prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            As an e-commerce support expert, create a helpful response using these guidelines:
            
            1. CONTEXT:
            {context}
            
            2. QUESTION: 
            {question}
            
            3. RESPONSE REQUIREMENTS:
            - Answer directly and precisely
            - Use bullet points for steps
            - Include examples where helpful
            - Add warning notes if applicable
            - Keep tone professional but friendly
            - Never invent information
            
            FINAL ANSWER:
            """
        )

    def generate_response(self, query: str) -> Dict[str, any]:
        """Generate context-aware response with semantic search"""
        try:
            # Handle greetings
            if self._is_greeting(query):
                return {
                    "answer": random.choice(self.greeting_responses),
                    "sources": []
                }
            
            # Generate embedding and search
            query_embedding = self.embedder.generate_embedding(query)
            results = self.vector_db.search(
                query_embedding=query_embedding,
                query=query,
                mode="semantic",
                threshold=0.65,
                limit=3
            )
            
            # Handle no results
            if not results:
                return {
                    "answer": random.choice(self.fallback_responses),
                    "sources": []
                }
            
            # Prepare enhanced context
            context = self._format_context(results, query)
            
            # Generate polished response
            response = self._generate_llm_response(query, context)
            
            return {
                "answer": response,
                "sources": results
            }
            
        except Exception as e:
            return {
                "answer": f"⚠️ Sorry, I encountered an error. Please try again later.\n(Error: {str(e)})",
                "sources": []
            }

    def _format_context(self, results: List[Dict], query: str) -> str:
        """Format search results into LLM context"""
        context_lines = []
        for i, res in enumerate(results, 1):
            context_lines.append(
                f"MATCH #{i} (Relevance: {res['score']:.0%}):\n"
                f"QUESTION: {res['text']}\n"
                f"ANSWER: {res['metadata'].get('answer', 'No specific answer found')}\n"
                f"SOURCE: {res['metadata'].get('source', 'General Knowledge')}\n"
            )
        return "\n".join(context_lines)

    def _generate_llm_response(self, query: str, context: str) -> str:
        """Generate final response using LLM"""
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run({
            "question": query,
            "context": context
        })

    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting phrase"""
        greetings = [
            "hello", "hi", "hey", "greetings", 
            "good morning", "good afternoon", "good evening",
            "namaste", "salam", "hola", "hi there", "helloo"
        ]
        text = text.lower().strip(" ?!.,")
        return any(greet in text for greet in greetings)

    def _get_helpful_links(self) -> str:
        """Generate dynamic help links"""
        return (
            "\n\nHelpful Links:\n"
            "• [Password Reset](https://example.com/reset)\n"
            "• [Order Tracking](https://example.com/track)\n"
            "• [Contact Support](mailto:support@example.com)"
        )