import os
import torch
import yaml
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from data_ingester import ChatbotDataIngester
from data_query import ChatbotDataQuery
from getpass import getpass
from pinecone import Pinecone, ServerlessSpec
from ragatouille import RAGPretrainedModel

import torch.nn.functional as F
from transformers import AutoModel

class CustomReranker:
    def __init__(self, model_name="nvidia/NV-Embed-v2", max_length=32768):
        """
        Initialize the reranker with the model and tokenizer.
        """
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.max_length = max_length

    def _encode(self, texts, instruction=""):
        """
        Helper function to encode the input texts using the model.
        """
        return self.model.encode(texts, instruction=instruction, max_length=self.max_length)

    def rerank(self, query, passages, k=1):
        """
        Rerank the passages based on their similarity with the query.
        
        Args:
        - query (str): The query text.
        - passages (list of str): List of passages to rerank.
        - k (int): The number of top-k documents to return after reranking.

        Returns:
        - A list of the top-k ranked passages with their similarity scores.
        """
        query_prefix = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
        passage_prefix = ""
        
        # Get the query and passage embeddings
        query_embeddings = self._encode([query], instruction=query_prefix)
        passage_embeddings = self._encode(passages, instruction=passage_prefix)
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
        
        # Compute similarity scores
        scores = (query_embeddings @ passage_embeddings.T) * 100
        scores = scores.tolist()[0]
        
        # Sort passages by their scores
        sorted_passages = sorted(
            [{"content": passage, "score": score, "result_index": idx}
             for idx, (passage, score) in enumerate(zip(passages, scores))],
            key=lambda x: x['score'], reverse=True
        )
        
        return sorted_passages[:k]  # Return top-k reranked passages

class RAGChatbot:
    def __init__(self, pinecone_api_key=None, index_name="test-index", config_path="../config.yml"):
        """
        Initialize the RAGChatbot. Handles embeddings, vector store, data ingestion, and query.
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")# or getpass("Enter your Pinecone API key: ")
        self.index_name = index_name
        self.embeddings = self.initialize_embeddings()
        self.dimensions = len(self.embeddings.embed_query("Hello World!"))
        self.vector_store = self.initialize_vector_store()
        self.data_ingester = ChatbotDataIngester(vector_store=self.vector_store, embeddings=self.embeddings)
        self.data_query = ChatbotDataQuery(vector_store=self.vector_store)
        self.reranker = self.initialize_reranker()
        # self.reranker = CustomReranker()
        
    def load_config(self, config_path):
        """
        Load the configuration file (config.yml).
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def initialize_embeddings(self):
        """
        Initialize the embedding model based on the config file.
        """
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        return hf

    def initialize_reranker(self):
        """
        Initialize the reranker
        """
        return RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def initialize_vector_store(self):
        """
        Initialize Pinecone vector store.
        """
        pc = Pinecone(api_key=self.pinecone_api_key)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if self.index_name not in existing_indexes:
            pc.create_index(
                name=self.index_name,
                dimension=self.dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(self.index_name).status["ready"]:
                import time
                time.sleep(1)

        return PineconeVectorStore(index=pc.Index(self.index_name), embedding=self.embeddings)

    def ingest_data(self, dir_path, empty=False):
        """
        Ingest data from a directory using the ChatbotDataIngester.
        """
        self.data_ingester.load_and_ingest(dir_path, empty_db=empty)

    def query_chatbot(self, query_text, k=1, rerank=False): #, fetch_k=2, lambda_mult=0.5
        """
        Query the chatbot using the provided query text and optional search parameters.
        """
        if rerank:
            response = self.data_query.query(
                query_text=query_text,
                k=k,
                reranker=self.reranker
            )
        else:
            response = self.data_query.query(
                query_text=query_text,
                k=k,
            )
        return response